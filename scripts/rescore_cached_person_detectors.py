"""Recompute detector scores from cached boxes without repeating inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_person_detectors import gold_frames, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    frames = list(range(0, args.frames, args.stride))
    truth = gold_frames(args.gold, set(frames))
    data = {frame: [] for frame in frames}
    with args.cache.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame not in data:
                continue
            data[frame].append({
                "box": np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2")]),
                "confidence": float(row["confidence"]),
            })
    rows = score(args.name, data, truth, frames)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cache": str(args.cache), "output": str(args.output),
                      "rows": len(rows)}))


if __name__ == "__main__":
    main()
