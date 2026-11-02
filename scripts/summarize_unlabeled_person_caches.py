"""Summarize proposal volumes by box height; never interpret as accuracy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_person_detectors import HEIGHT_BINS, THRESHOLDS


def parse_candidate(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError
        return name, Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use name=cache.csv") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--sampled-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    result = []
    for name, path in args.candidate:
        with path.open(newline="", encoding="utf-8") as handle:
            detections = list(csv.DictReader(handle))
        for threshold in THRESHOLDS:
            selected = [row for row in detections if float(row["confidence"]) >= threshold]
            by_height = {}
            for label, low, high in HEIGHT_BINS:
                count = sum(low <= float(row["y2"]) - float(row["y1"]) < high
                            for row in selected)
                by_height[label] = count
            result.append({
                "candidate": name, "confidence": threshold,
                "total_boxes": len(selected),
                "boxes_per_frame": len(selected) / args.sampled_frames,
                **by_height,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
