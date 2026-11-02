"""Cache one fixed YOLO11l detection pass for the Aria rotation ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    if not args.video.is_file() or not args.model.is_file():
        parser.error("video and model must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "yolo11l_detections.csv"
    if output.exists():
        parser.error(f"Refusing to overwrite existing detection cache: {output}")

    detector = YOLO(str(args.model))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fields = ("frame", "detection_id", "x1", "y1", "x2", "y2", "confidence")
    frames = detections = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(args.frames):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {frame}; requested {args.frames}")
            result = detector.predict(
                image,
                classes=[0],
                imgsz=1280,
                conf=0.15,
                iou=0.70,
                device=0,
                half=True,
                verbose=False,
            )[0]
            for detection_id, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                writer.writerow(
                    {
                        "frame": frame,
                        "detection_id": detection_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": float(box.conf[0]),
                    }
                )
                detections += 1
            frames += 1
            if frames % 50 == 0:
                print(f"frames={frames} detections={detections}", flush=True)
    capture.release()
    (args.output_dir / "yolo11l_detection_manifest.json").write_text(
        json.dumps(
            {
                "video": str(args.video.resolve()),
                "video_sha256": sha256(args.video),
                "model": str(args.model.resolve()),
                "model_sha256": sha256(args.model),
                "frames": frames,
                "detections": detections,
                "imgsz": 1280,
                "conf": 0.15,
                "nms_iou": 0.70,
                "classes": [0],
                "device": "cuda:0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"complete frames={frames} detections={detections} output={output}")


if __name__ == "__main__":
    main()
