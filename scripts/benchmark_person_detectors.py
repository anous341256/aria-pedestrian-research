"""Detection-only person-model screen against reviewed Aria boxes.

The gold is not assumed exhaustive for tiny pedestrians. Unmatched proposals
are reported as review candidates, not automatically as false positives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from audit_cvat_video_annotations import parse_xml  # noqa: E402


HEIGHT_BINS = (("h_lt_48", 0, 48), ("h_48_96", 48, 96),
               ("h_96_192", 96, 192), ("h_ge_192", 192, float("inf")))
THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path, int]:
    try:
        name, rest = value.split("=", 1)
        path, size = rest.rsplit("@", 1)
        if not name or not path or int(size) <= 0:
            raise ValueError
        return name, Path(path), int(size)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use --candidate name=path/to/weights.pt@imgsz"
        ) from error


def gold_frames(path: Path, frame_ids: set[int]) -> dict[int, list[np.ndarray]]:
    parsed = parse_xml(path)
    result = {frame: [] for frame in frame_ids}
    for track in parsed["tracks"].values():
        if track["label"] not in {"", "person"}:
            continue
        for frame, box in track["visible"].items():
            if frame in result:
                result[frame].append(np.asarray(box, dtype=np.float64))
    return result


def iou_matrix(gold: list[np.ndarray], predicted: list[np.ndarray]) -> np.ndarray:
    if not gold or not predicted:
        return np.zeros((len(gold), len(predicted)), dtype=np.float64)
    a = np.stack(gold)
    b = np.stack(predicted)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def match(gold: list[np.ndarray], predicted: list[np.ndarray], threshold: float = 0.5
          ) -> tuple[set[int], set[int]]:
    similarity = iou_matrix(gold, predicted)
    if not similarity.size:
        return set(), set()
    # Maximize the number of IoU-valid matches first, then their summed IoU.
    # Plain (1 - IoU) assignment can sacrifice one valid match to obtain a
    # larger IoU on another pair, biasing recall in crowded frames.
    cost = np.where(similarity >= threshold, -(1 + similarity), 0.0)
    rows, cols = linear_sum_assignment(cost)
    accepted = [(int(i), int(j)) for i, j in zip(rows, cols)
                if similarity[i, j] >= threshold]
    return {i for i, _ in accepted}, {j for _, j in accepted}


def score(name: str, data: dict[int, list[dict]], truth: dict[int, list[np.ndarray]],
          selected_frames: list[int]) -> list[dict]:
    rows = []
    for confidence in THRESHOLDS:
        totals = {"all": 0, **{label: 0 for label, _, _ in HEIGHT_BINS}}
        found = {key: 0 for key in totals}
        proposals = unmatched = frames_with_unmatched = 0
        for frame in selected_frames:
            reference = truth[frame]
            candidates = [row["box"] for row in data[frame]
                          if row["confidence"] >= confidence]
            matched_gt, matched_pred = match(reference, candidates)
            totals["all"] += len(reference)
            found["all"] += len(matched_gt)
            proposals += len(candidates)
            unmatched += len(candidates) - len(matched_pred)
            frames_with_unmatched += int(len(candidates) > len(matched_pred))
            for index, box in enumerate(reference):
                height = box[3] - box[1]
                for label, low, high in HEIGHT_BINS:
                    if low <= height < high:
                        totals[label] += 1
                        found[label] += index in matched_gt
                        break
        row = {"candidate": name, "confidence": confidence,
               "sampled_frames": len(selected_frames), "gold_boxes": totals["all"],
               "matched_gold": found["all"],
               "recall_all": found["all"] / max(totals["all"], 1),
               "proposals_per_frame": proposals / len(selected_frames),
               "unmatched_proposals_per_frame": unmatched / len(selected_frames),
               "frames_with_unmatched_fraction": frames_with_unmatched / len(selected_frames)}
        for label, _, _ in HEIGHT_BINS:
            row[f"gold_{label}"] = totals[label]
            row[f"matched_{label}"] = found[label]
            row[f"recall_{label}"] = found[label] / max(totals[label], 1)
        rows.append(row)
    return rows


def predict_candidate(name: str, weights: Path, imgsz: int, video: Path,
                      selected_frames: list[int], output_dir: Path) -> tuple[dict[int, list[dict]], dict]:
    cache = output_dir / f"detections_{name}.csv"
    if cache.exists():
        raise FileExistsError(f"Refusing to overwrite existing detection cache: {cache}")
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    data = {frame: [] for frame in selected_frames}
    latencies = []
    fields = ("frame", "detection_id", "x1", "y1", "x2", "y2", "confidence")
    try:
        with cache.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, frame in enumerate(selected_frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read video frame {frame}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = model.predict(
                    image, classes=[0], imgsz=imgsz, conf=0.05, iou=0.70,
                    device=0, half=True, max_det=300, verbose=False,
                )[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                if index >= 5:
                    latencies.append(elapsed_ms)
                for detection_id, box in enumerate(result.boxes):
                    coordinates = box.xyxy[0].cpu().numpy().astype(float)
                    confidence = float(box.conf[0])
                    data[frame].append({"box": coordinates, "confidence": confidence})
                    writer.writerow({"frame": frame, "detection_id": detection_id,
                                     **dict(zip(("x1", "y1", "x2", "y2"), coordinates)),
                                     "confidence": confidence})
                if (index + 1) % 50 == 0:
                    print(f"{name}: {index + 1}/{len(selected_frames)} frames", flush=True)
    finally:
        capture.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return data, {"name": name, "weights": str(weights.resolve()),
                  "weights_sha256": sha256(weights), "imgsz": imgsz,
                  "inference_median_ms": statistics.median(latencies) if latencies else None,
                  "detections": sum(len(rows) for rows in data.values()),
                  "cache": str(cache.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gold", type=Path,
                        help="Reviewed CVAT XML; omit for unlabeled proposal generation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    selected_frames = list(range(0, args.frames, args.stride))
    truth = gold_frames(args.gold, set(selected_frames)) if args.gold else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    manifests = []
    for name, weights, imgsz in args.candidate:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        data, manifest = predict_candidate(
            name, weights, imgsz, args.video, selected_frames, args.output_dir
        )
        if truth is not None:
            all_scores.extend(score(name, data, truth, selected_frames))
        manifests.append(manifest)
    if all_scores:
        with (args.output_dir / "detection_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_scores[0]))
            writer.writeheader()
            writer.writerows(all_scores)
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps({"video": str(args.video.resolve()), "video_sha256": sha256(args.video),
                    "gold": str(args.gold.resolve()) if args.gold else None,
                    "gold_sha256": sha256(args.gold) if args.gold else None,
                    "frames": selected_frames, "iou_threshold": 0.5,
                    "confidence_thresholds": THRESHOLDS, "candidates": manifests,
                    "caveat": "Unmatched proposals may be unlabeled real distant people; gold is not exhaustive."},
                   indent=2), encoding="utf-8"
    )
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
