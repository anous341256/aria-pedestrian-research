"""Run the same Aria detection-only screen with official RF-DETR weights.

Use the isolated E-drive rfdetr-benchmark environment. Published checkpoint
downloads are controlled by RF_HOME and should also be set to an E-drive path.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import rfdetr
import torch

from benchmark_person_detectors import gold_frames, score, sha256


MODELS = {
    "medium": rfdetr.RFDETRMedium,
    "large": rfdetr.RFDETRLarge,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gold", type=Path,
                        help="Reviewed CVAT XML; omit for unlabeled proposal generation")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=MODELS, default="large")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    selected = list(range(0, args.frames, args.stride))
    truth = gold_frames(args.gold, set(selected)) if args.gold else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"rfdetr_{args.model}"
    cache = args.output_dir / f"detections_{name}.csv"
    if cache.exists():
        raise FileExistsError(f"Refusing to overwrite existing detection cache: {cache}")

    model = MODELS[args.model]()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    data = {frame: [] for frame in selected}
    latency = []
    fields = ("frame", "detection_id", "x1", "y1", "x2", "y2", "confidence")
    try:
        with cache.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, frame in enumerate(selected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read video frame {frame}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                detections = model.predict(
                    image[:, :, ::-1].copy(), threshold=0.05,
                    include_source_image=False,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if index >= 5:
                    latency.append((time.perf_counter() - start) * 1000)
                # Official COCO checkpoints retain sparse COCO category ID 1 for person.
                person_indices = np.flatnonzero(detections.class_id == 1)
                for detection_id, j in enumerate(person_indices):
                    box = detections.xyxy[j].astype(float)
                    confidence = float(detections.confidence[j])
                    data[frame].append({"box": box, "confidence": confidence})
                    writer.writerow({"frame": frame, "detection_id": detection_id,
                                     **dict(zip(("x1", "y1", "x2", "y2"), box)),
                                     "confidence": confidence})
                if (index + 1) % 25 == 0:
                    print(f"{name}: {index + 1}/{len(selected)} frames", flush=True)
    finally:
        capture.release()

    if truth is not None:
        rows = score(name, data, truth, selected)
        with (args.output_dir / "detection_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps({
            "video": str(args.video.resolve()), "video_sha256": sha256(args.video),
            "gold": str(args.gold.resolve()) if args.gold else None,
            "gold_sha256": sha256(args.gold) if args.gold else None,
            "frames": selected, "iou_threshold": 0.5,
            "model": name, "rfdetr_version": rfdetr.__version__ if hasattr(rfdetr, "__version__") else None,
            "torch_version": torch.__version__,
            "inference_median_ms": statistics.median(latency) if latency else None,
            "detections": sum(len(x) for x in data.values()),
            "caveat": "Unmatched proposals may be unlabeled real distant people; gold is not exhaustive.",
        }, indent=2), encoding="utf-8"
    )
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
