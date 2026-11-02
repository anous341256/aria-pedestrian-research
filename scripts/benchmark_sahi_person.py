"""Run YOLO + SAHI tiled inference on the reviewed Aria segment.

The detector sees 1280-pixel tiles and a full-image pass; all predictions are
merged by SAHI before applying the same gold-box scorer as the other models.
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
import sahi
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from benchmark_person_detectors import gold_frames, score, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gold", type=Path,
                        help="Reviewed CVAT XML; omit for unlabeled proposal generation")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--tile", type=int, default=1280)
    args = parser.parse_args()
    selected = list(range(0, args.frames, args.stride))
    truth = gold_frames(args.gold, set(selected)) if args.gold else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"yolo11l_sahi_{args.tile}"
    cache = args.output_dir / f"detections_{name}.csv"
    if cache.exists():
        raise FileExistsError(f"Refusing to overwrite existing detection cache: {cache}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(args.weights),
        confidence_threshold=0.05, image_size=args.tile, device="cuda:0",
    )
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
                result = get_sliced_prediction(
                    image[:, :, ::-1].copy(), model,
                    slice_height=args.tile, slice_width=args.tile,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                    perform_standard_pred=True,
                    postprocess_type="NMS", postprocess_match_metric="IOU",
                    postprocess_match_threshold=0.5,
                    verbose=0,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if index >= 5:
                    latency.append((time.perf_counter() - start) * 1000)
                for detection_id, obj in enumerate(result.object_prediction_list):
                    if obj.category.id != 0:
                        continue
                    box = np.asarray(obj.bbox.to_xyxy(), dtype=float)
                    confidence = float(obj.score.value)
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
            "model": name, "weights": str(args.weights.resolve()),
            "weights_sha256": sha256(args.weights), "sahi_version": sahi.__version__,
            "tile": args.tile, "tile_overlap": 0.2, "full_image_pass": True,
            "merge": "NMS/IOU/0.5",
            "inference_median_ms": statistics.median(latency) if latency else None,
            "detections": sum(len(x) for x in data.values()),
            "caveat": "Unmatched proposals may be unlabeled real distant people; gold is not exhaustive.",
        }, indent=2), encoding="utf-8"
    )
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
