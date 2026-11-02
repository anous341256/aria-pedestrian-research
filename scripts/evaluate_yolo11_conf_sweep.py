"""Compare fixed-tracker CVAT proposals from a YOLO confidence sweep.

Gold may omit distant pedestrians, so unmatched predictions are review
candidates rather than verified false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from audit_cvat_video_annotations import parse_xml  # noqa: E402
from benchmark_person_detectors import HEIGHT_BINS, gold_frames, match  # noqa: E402


def visible_frames(path: Path, frame_ids: set[int]) -> dict[int, list[np.ndarray]]:
    parsed = parse_xml(path)
    result = {frame: [] for frame in frame_ids}
    for track in parsed["tracks"].values():
        if track["label"] not in {"", "person"}:
            continue
        for frame, box in track["visible"].items():
            if frame in result:
                result[frame].append(np.asarray(box, dtype=np.float64))
    return result


def height_bin(box: np.ndarray) -> str:
    height = float(box[3] - box[1])
    return next(label for label, low, high in HEIGHT_BINS if low <= height < high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lower", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()

    frame_ids = list(range(args.frames))
    frame_set = set(frame_ids)
    truth = gold_frames(args.gold, frame_set)
    paths = {"conf_015": args.baseline}
    for path in args.lower:
        name = path.parent.name.replace("conf_sweep_yolo11l_", "conf_", 1)
        if name in paths:
            raise ValueError(f"Duplicate condition: {name}")
        paths[name] = path
    predictions = {name: visible_frames(path, frame_set) for name, path in paths.items()}

    conditions = {}
    for name, sequence in predictions.items():
        matched_by_frame = {}
        counts = {"all": {"gold": 0, "matched": 0}}
        counts.update({label: {"gold": 0, "matched": 0} for label, _, _ in HEIGHT_BINS})
        proposals = unmatched = 0
        for frame in frame_ids:
            gold = truth[frame]
            proposed = sequence[frame]
            matched_gold, matched_proposed = match(gold, proposed)
            matched_by_frame[frame] = matched_gold
            proposals += len(proposed)
            unmatched += len(proposed) - len(matched_proposed)
            counts["all"]["gold"] += len(gold)
            counts["all"]["matched"] += len(matched_gold)
            for index, box in enumerate(gold):
                bucket = height_bin(box)
                counts[bucket]["gold"] += 1
                counts[bucket]["matched"] += int(index in matched_gold)
        for item in counts.values():
            item["recall"] = item["matched"] / item["gold"] if item["gold"] else None
        conditions[name] = {
            "source": str(paths[name].resolve()),
            "counts": counts,
            "proposals": proposals,
            "proposals_per_frame": proposals / args.frames,
            "unmatched_review_candidates": unmatched,
            "unmatched_review_candidates_per_frame": unmatched / args.frames,
            "matched_by_frame": matched_by_frame,
        }

    baseline_matches = conditions["conf_015"]["matched_by_frame"]
    changes = {}
    detail = []
    for name, result in conditions.items():
        if name == "conf_015":
            continue
        bins = {"all": {"recovered": 0, "regressed": 0}}
        bins.update({label: {"recovered": 0, "regressed": 0} for label, _, _ in HEIGHT_BINS})
        for frame in frame_ids:
            for index, box in enumerate(truth[frame]):
                before = index in baseline_matches[frame]
                after = index in result["matched_by_frame"][frame]
                if before == after:
                    continue
                change = "recovered" if after else "regressed"
                bucket = height_bin(box)
                bins["all"][change] += 1
                bins[bucket][change] += 1
                detail.append({"condition": name, "frame": frame,
                               "gold_index": index, "change": change,
                               "height_bin": bucket, "gold_height": float(box[3] - box[1]),
                               "x1": float(box[0]), "y1": float(box[1]),
                               "x2": float(box[2]), "y2": float(box[3])})
        changes[name] = bins

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if detail:
        with (args.output_dir / "changed_gold_boxes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
            writer.writeheader()
            writer.writerows(detail)
    for result in conditions.values():
        del result["matched_by_frame"]
    report = {
        "gold": str(args.gold.resolve()),
        "frames": args.frames,
        "matching_iou": 0.5,
        "conditions": conditions,
        "paired_changes_vs_conf_015": changes,
        "caveat": "Gold omits some distant people; unmatched proposals are not verified false positives.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
