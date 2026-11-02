"""One-pass YOLO detection with parallel current and Deep OC-SORT tracking.

The callback captures raw post-NMS YOLO detections before Ultralytics mutates
the result into tracks.  The exact same detections are cached and supplied to
BoxMOT Deep OC-SORT.  This guarantees one detector pass and comparable tracker
inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from boxmot import ReIDConfig, create_tracker
from ultralytics import YOLO

from auto_annotate_people import video_metadata, write_cvat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--current-tracker", type=Path, required=True)
    parser.add_argument("--deepoc-reid", default="osnet-x0-25-msmt17")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def add_track_row(store: dict[int, list[dict]], frame_id: int, output) -> None:
    x1, y1, x2, y2, track_id, confidence, class_id = output[:7]
    if int(class_id) != 0:
        return
    store[int(track_id)].append(
        {
            "frame_id": frame_id,
            "track_id": int(track_id),
            "label": "person",
            "confidence": float(confidence),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "footpoint": [float((x1 + x2) / 2.0), float(y2)],
        }
    )


def write_mot(path: Path, tracks: dict[int, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for track_id, rows in sorted(tracks.items()):
        for row in rows:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            lines.append(
                f"{row['frame_id'] + 1},{track_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{row['confidence']:.6f},1,1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_proposal(
    root: Path,
    name: str,
    tracks: dict[int, list[dict]],
    video: Path,
    metadata: dict,
) -> dict:
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    write_cvat(output / "annotations.xml", tracks, video, metadata)
    write_mot(output / "gt" / "gt.txt", tracks)
    summary = {
        "track_count": len(tracks),
        "box_count": sum(len(rows) for rows in tracks.values()),
        "tracks_at_least_60_frames": sum(len(rows) >= 60 for rows in tracks.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    for path in (args.video, args.model, args.current_tracker):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    reid = ReIDConfig(
        args.deepoc_reid,
        device=device,
        precision="fp16" if device.startswith("cuda") else "fp32",
    )
    deepoc = create_tracker(
        "deepocsort", reid=reid, class_ids=(0,), class_names={0: "person"}
    )
    model = YOLO(str(args.model))

    detection_rows: list[dict] = []
    deepoc_tracks: dict[int, list[dict]] = defaultdict(list)
    current_tracks: dict[int, list[dict]] = defaultdict(list)
    pending_detections: list[np.ndarray] = []
    pending_images: list[np.ndarray] = []
    callback_frame = 0
    deepoc_seconds = 0.0

    def capture_raw_and_run_deepoc(predictor) -> None:
        nonlocal callback_frame, deepoc_seconds
        for result in predictor.results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections = np.empty((0, 6), dtype=np.float32)
            else:
                detections = np.column_stack(
                    (
                        boxes.xyxy.detach().cpu().numpy(),
                        boxes.conf.detach().cpu().numpy(),
                        boxes.cls.detach().cpu().numpy(),
                    )
                ).astype(np.float32, copy=False)
            for detection_id, row in enumerate(detections):
                x1, y1, x2, y2, confidence, class_id = row
                detection_rows.append(
                    {
                        "frame_id": callback_frame,
                        "detection_id": detection_id,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                    }
                )
            # Do not execute the second tracker inside this callback. Some
            # trackers/GMC implementations touch image state in-place, which
            # can perturb the built-in BoT-SORT callback that runs next.
            pending_detections.append(detections.copy())
            pending_images.append(result.orig_img.copy())
            callback_frame += 1

    # model.track registers its own tracking callback after this callback, so
    # this function sees the raw post-NMS detections before IDs are attached.
    model.add_callback("on_predict_postprocess_end", capture_raw_and_run_deepoc)

    started = time.perf_counter()
    stream = model.track(
        source=str(args.video),
        stream=True,
        persist=True,
        classes=[0],
        tracker=str(args.current_tracker),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    processed = 0
    for frame_id, result in enumerate(stream):
        if args.max_frames is not None and frame_id >= args.max_frames:
            break
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            for track_id, xyxy, confidence, class_id in zip(ids, coords, confs, classes):
                add_track_row(
                    current_tracks,
                    frame_id,
                    [*xyxy, track_id, confidence, class_id],
                )
        # BoT-SORT has now consumed this frame. Feed the exact cached post-NMS
        # detections to Deep OC-SORT without invoking YOLO a second time.
        tick = time.perf_counter()
        outputs = deepoc.update(pending_detections[frame_id], pending_images[frame_id])
        deepoc_seconds += time.perf_counter() - tick
        for output in outputs:
            add_track_row(deepoc_tracks, frame_id, output)
        processed = frame_id + 1
        if processed == 1 or processed % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {"frames": processed, "elapsed_s": round(elapsed, 1), "fps": round(processed / elapsed, 2)}
                ),
                flush=True,
            )

    # The callback may observe one look-ahead frame before the consumer stops.
    detection_rows = [row for row in detection_rows if row["frame_id"] < processed]
    deepoc_tracks = defaultdict(
        list,
        {
            track_id: [row for row in rows if row["frame_id"] < processed]
            for track_id, rows in deepoc_tracks.items()
            if any(row["frame_id"] < processed for row in rows)
        },
    )
    metadata = video_metadata(args.video)
    metadata["processed_frames"] = processed

    pd.DataFrame(
        detection_rows,
        columns=["frame_id", "detection_id", "x1", "y1", "x2", "y2", "confidence", "class_id"],
    ).to_parquet(args.output_dir / "detections.parquet", index=False)
    summaries = {
        "current_raw": write_proposal(args.output_dir, "current_raw", current_tracks, args.video, metadata),
        "deepocsort": write_proposal(args.output_dir, "deepocsort", deepoc_tracks, args.video, metadata),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": str(args.video.resolve()),
        "video_sha256": sha256(args.video),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "current_tracker": str(args.current_tracker.resolve()),
        "deepoc_reid": args.deepoc_reid,
        "processed_frames": processed,
        "raw_detection_count": len(detection_rows),
        "settings": {
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "boxmot": __import__("boxmot").__version__,
            "ultralytics": __import__("ultralytics").__version__,
        },
        "deepoc_update_seconds": deepoc_seconds,
        "total_seconds": time.perf_counter() - started,
        "summaries": summaries,
    }
    (args.output_dir / "proposal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
