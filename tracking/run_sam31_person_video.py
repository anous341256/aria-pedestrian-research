"""Run Meta SAM 3.1 on a video and export person tracks for CVAT/MOT evaluation.

The text prompt and threshold are fixed before looking at the ground truth.
SAM's normalized mask-derived boxes are converted to original video pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from auto_annotate_people import video_metadata, write_cvat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-objects", type=int, default=16)
    return parser.parse_args()


def extract_rows(frame_id: int, output: dict, width: int, height: int) -> list[dict]:
    ids = output["out_obj_ids"]
    boxes = output["out_boxes_xywh"]
    probabilities = output["out_probs"]
    if not (len(ids) == len(boxes) == len(probabilities)):
        raise ValueError(f"SAM output lengths disagree at frame {frame_id}")

    rows = []
    for object_id, box, probability in zip(ids, boxes, probabilities):
        x, y, w, h = (float(value) for value in box)
        x1 = max(0.0, min(width, x * width))
        y1 = max(0.0, min(height, y * height))
        x2 = max(0.0, min(width, (x + w) * width))
        y2 = max(0.0, min(height, (y + h) * height))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "track_id": int(object_id),
                "label": "person",
                "confidence": float(probability),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie in [0, 1]")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")

    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 comparison requires the CUDA GPU")

    meta = video_metadata(args.video)
    limit = min(meta["frame_count"], args.max_frames or meta["frame_count"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(args.checkpoint),
        max_num_objects=args.max_objects,
        use_fa3=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        default_output_prob_thresh=args.threshold,
    )
    print("Model loaded; starting video session", flush=True)
    session_id = None
    frame_rows: dict[int, list[dict]] = {}
    try:
        session_id = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(args.video),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )["session_id"]
        first = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.prompt,
                "output_prob_thresh": args.threshold,
            }
        )
        frame_rows[0] = extract_rows(0, first["outputs"], meta["width"], meta["height"])

        with (args.output_dir / "raw_predictions.jsonl").open("w", encoding="utf-8") as raw:
            for response in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "max_frame_num_to_track": limit,
                    "output_prob_thresh": args.threshold,
                    "evict_cached_frame_outputs": True,
                }
            ):
                frame_id = int(response["frame_index"])
                if frame_id >= limit:
                    break
                rows = extract_rows(
                    frame_id, response["outputs"], meta["width"], meta["height"]
                )
                frame_rows[frame_id] = rows
                raw.write(json.dumps({"frame": frame_id, "rows": rows}) + "\n")
                raw.flush()
                if (frame_id + 1) % 25 == 0:
                    print(f"Processed {frame_id + 1}/{limit} frames", flush=True)
    finally:
        if session_id is not None:
            predictor.handle_request({"type": "close_session", "session_id": session_id})

    tracks: dict[int, list[dict]] = defaultdict(list)
    mot_lines = []
    for frame_id in range(limit):
        for row in frame_rows.get(frame_id, []):
            tracks[row["track_id"]].append(row)
            x1, y1, x2, y2 = row["bbox_xyxy"]
            mot_lines.append(
                f'{frame_id + 1},{row["track_id"]},{x1:.2f},{y1:.2f},'
                f'{x2 - x1:.2f},{y2 - y1:.2f},{row["confidence"]:.6f},1,1'
            )

    meta["processed_frames"] = limit
    write_cvat(args.output_dir / "annotations.xml", tracks, args.video, meta)
    (args.output_dir / "prediction_mot.txt").write_text(
        "\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8"
    )
    summary = {
        "model": "Meta SAM 3.1 Object Multiplex",
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "video": str(args.video),
        "prompt": args.prompt,
        "threshold": args.threshold,
        "max_objects": args.max_objects,
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "use_fa3": False,
        "processed_frames": limit,
        "frames_with_response": len(frame_rows),
        "tracks": len(tracks),
        "boxes": len(mot_lines),
        "runtime_seconds": round(time.monotonic() - started, 2),
        "gpu": torch.cuda.get_device_name(0),
        "warning": "Automatic proposals, not ground truth.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
