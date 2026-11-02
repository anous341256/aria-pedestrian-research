"""Measure gold-box complementarity of two cached person detectors.

This is a coverage upper bound before proposal deduplication or tracking, not
an evaluated ensemble pipeline.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark_person_detectors import HEIGHT_BINS, gold_frames, match


def read_cache(path: Path, threshold: float) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["confidence"]) >= threshold:
                result[int(row["frame"])].append(
                    np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2")])
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-conf", type=float, default=0.05)
    parser.add_argument("--second-conf", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    selected = list(range(0, args.frames, args.stride))
    truth = gold_frames(args.gold, set(selected))
    first = read_cache(args.first, args.first_conf)
    second = read_cache(args.second, args.second_conf)
    totals = {"all": [0, 0, 0, 0], **{name: [0, 0, 0, 0] for name, _, _ in HEIGHT_BINS}}
    for frame in selected:
        boxes = truth[frame]
        one, _ = match(boxes, first[frame])
        two, _ = match(boxes, second[frame])
        for index, box in enumerate(boxes):
            height = box[3] - box[1]
            name = next(name for name, lo, hi in HEIGHT_BINS if lo <= height < hi)
            for key in ("all", name):
                totals[key][0] += 1
                totals[key][1] += index in one
                totals[key][2] += index in two
                totals[key][3] += index in one or index in two
    for key, (n, a, b, both) in totals.items():
        print(f"{key}: gold={n}, first={a}, second={b}, union={both}, "
              f"extra_over_first={both-a}")


if __name__ == "__main__":
    main()
