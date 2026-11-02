"""Run a fair BoxMOT tracker comparison on one video.

YOLO detections are computed once per frame and fed unchanged to every
tracker.  Each candidate is exported as CVAT video XML and MOTChallenge text
so that ``evaluate_cvat_mot.py`` can score it against the corrected reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import ReIDConfig, create_tracker
from ultralytics import YOLO

from auto_annotate_people import video_metadata, write_cvat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--reid",
        default="osnet-x0-25-msmt17",
        help="BoxMOT ReID registry name or a supported local artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["botsort", "bytetrack", "deepocsort"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--profile",
        choices=("official", "aria"),
        default="official",
        help="Official BoxMOT defaults or confidence/buffer settings matched to the Aria baseline.",
    )
    return parser.parse_args()


def make_tracker(name: str, reid_model: str, device: str, profile: str):
    reid = None
    if name in {"botsort", "deepocsort", "strongsort", "boosttrack"}:
        candidate = Path(reid_model)
        model = candidate.resolve() if candidate.is_file() else reid_model
        reid = ReIDConfig(
            model,
            device=device,
            precision="fp16" if device.startswith("cuda") else "fp32",
            allow_download=not candidate.is_file(),
        )
    overrides = {}
    if profile == "aria":
        overrides = {
            "botsort": {
                "det_thresh": 0.15,
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.35,
                "track_buffer": 90,
                "match_thresh": 0.80,
                "proximity_thresh": 0.50,
                "appearance_thresh": 0.80,
            },
            "bytetrack": {
                "track_thresh": 0.25,
                "min_conf": 0.10,
                "match_thresh": 0.80,
                "track_buffer": 90,
            },
            "deepocsort": {"det_thresh": 0.15},
        }.get(name, {})
    return create_tracker(
        name,
        reid=reid,
        class_ids=(0,),
        class_names={0: "person"},
        **overrides,
    )


def write_candidate(
    output_dir: Path,
    name: str,
    tracks: dict[int, list[dict]],
    video: Path,
    metadata: dict,
    elapsed_s: float,
) -> dict:
    candidate_dir = output_dir / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "gt").mkdir(exist_ok=True)
    write_cvat(candidate_dir / "annotations.xml", tracks, video, metadata)

    mot_lines = []
    for track_id, rows in sorted(tracks.items()):
        for row in rows:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            mot_lines.append(
                f"{row['frame_id'] + 1},{track_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{row['confidence']:.6f},1,1"
            )
    (candidate_dir / "gt" / "gt.txt").write_text(
        "\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8"
    )

    summary = {
        "tracker": name,
        "processed_frames": metadata["processed_frames"],
        "track_count": len(tracks),
        "box_count": sum(len(rows) for rows in tracks.values()),
        "tracks_at_least_60_frames": sum(len(rows) >= 60 for rows in tracks.values()),
        "tracker_update_seconds": elapsed_s,
    }
    (candidate_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    for path in (args.video, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(args.model))
    trackers = {
        name: make_tracker(name, args.reid, args.device, args.profile)
        for name in args.trackers
    }
    track_rows: dict[str, dict[int, list[dict]]] = {
        name: defaultdict(list) for name in trackers
    }
    tracker_seconds = {name: 0.0 for name in trackers}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    metadata = video_metadata(args.video)
    limit = min(metadata["frame_count"], args.max_frames)

    started = time.perf_counter()
    frame_id = 0
    while frame_id < limit:
        ok, frame = cap.read()
        if not ok:
            break
        prediction = detector.predict(
            frame,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        boxes = prediction.boxes
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

        for name, tracker in trackers.items():
            tick = time.perf_counter()
            outputs = tracker.update(detections.copy(), frame)
            tracker_seconds[name] += time.perf_counter() - tick
            for output in outputs:
                x1, y1, x2, y2, track_id, confidence, class_id, _ = output
                if int(class_id) != 0:
                    continue
                row = {
                    "frame_id": frame_id,
                    "track_id": int(track_id),
                    "label": "person",
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "footpoint": [float((x1 + x2) / 2.0), float(y2)],
                }
                track_rows[name][int(track_id)].append(row)

        frame_id += 1
        if frame_id == 1 or frame_id % 50 == 0 or frame_id == limit:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "frames": frame_id,
                        "limit": limit,
                        "elapsed_s": round(elapsed, 1),
                        "fps": round(frame_id / max(elapsed, 1e-9), 2),
                    }
                ),
                flush=True,
            )
    cap.release()
    metadata["processed_frames"] = frame_id

    summaries = [
        write_candidate(
            args.output_dir,
            name,
            track_rows[name],
            args.video,
            metadata,
            tracker_seconds[name],
        )
        for name in args.trackers
    ]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": str(args.video.resolve()),
        "detector": str(args.model.resolve()),
        "reid": args.reid,
        "detector_settings": {
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "nms_iou": args.iou,
            "device": args.device,
        },
        "tracker_profile": args.profile,
        "boxmot_version": __import__("boxmot").__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "total_seconds": time.perf_counter() - started,
        "summaries": summaries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
