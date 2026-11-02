"""Compare two detection caches on an unlabeled video without inventing accuracy.

Outputs per-frame agreement and unmatched candidate boxes for targeted review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark_person_detectors import HEIGHT_BINS, iou_matrix, match


def load(path: Path, confidence: float) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["confidence"]) >= confidence:
                result[int(row["frame"])].append({
                    "box": np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")]),
                    "confidence": float(row["confidence"]),
                })
    return result


def height_bin(box: np.ndarray) -> str:
    height = box[3] - box[1]
    return next(name for name, low, high in HEIGHT_BINS if low <= height < high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into nonempty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    one = load(args.first, args.confidence)
    two = load(args.second, args.confidence)
    per_frame = []
    exclusive = []
    totals = {args.first_name: defaultdict(int), args.second_name: defaultdict(int)}
    matched_total = 0
    for frame in range(0, args.frames, args.stride):
        first = one[frame]
        second = two[frame]
        matched_first, matched_second = match(
            [row["box"] for row in first], [row["box"] for row in second]
        )
        matched_total += len(matched_first)
        per_frame.append({"frame": frame, "first_count": len(first),
                          "second_count": len(second), "matched": len(matched_first),
                          "first_exclusive": len(first) - len(matched_first),
                          "second_exclusive": len(second) - len(matched_second)})
        matrix = iou_matrix([row["box"] for row in first],
                            [row["box"] for row in second])
        for name, rows, matched, axis in (
            (args.first_name, first, matched_first, 1),
            (args.second_name, second, matched_second, 0),
        ):
            for index, row in enumerate(rows):
                totals[name]["all"] += 1
                if index in matched:
                    totals[name]["matched"] += 1
                    continue
                label = height_bin(row["box"])
                totals[name]["exclusive"] += 1
                totals[name][label] += 1
                nearest = float(matrix[index, :].max()) if axis == 1 and matrix.shape[1] else 0.0
                if axis == 0:
                    nearest = float(matrix[:, index].max()) if matrix.shape[0] else 0.0
                exclusive.append({
                    "model": name, "frame": frame, "confidence": row["confidence"],
                    "x1": row["box"][0], "y1": row["box"][1],
                    "x2": row["box"][2], "y2": row["box"][3],
                    "height_bin": label, "nearest_other_iou": nearest,
                })
    for filename, rows in (("per_frame.csv", per_frame), ("exclusive_boxes.csv", exclusive)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "source": "unlabeled cross-recording proposal comparison; not precision or recall",
        "first": args.first_name, "second": args.second_name,
        "confidence": args.confidence, "iou_match": 0.5,
        "sampled_frames": len(per_frame), "matched_pairs": matched_total,
        "totals": {name: dict(values) for name, values in totals.items()},
        "mean_exclusive_per_frame": {
            args.first_name: sum(row["first_exclusive"] for row in per_frame) / len(per_frame),
            args.second_name: sum(row["second_exclusive"] for row in per_frame) / len(per_frame),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
