"""Audit fixed boxes and paired 20-frame-block uncertainty for the Aria experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from audit_cvat_video_annotations import parse_xml  # noqa: E402


METHODS = (
    "none", "rgb_flow", "legacy_pinhole", "aria_fisheye",
    "aria_plus_rgb_residual", "inverse_rotation", "inverse_plus_rgb_residual",
)
INPUTS = ("oracle_gold_boxes", "fixed_yolo11l_boxes")


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def signature(dataset: dict) -> tuple:
    return tuple(sorted((int(frame), *(round(float(value), 2) for value in box))
                        for track in dataset["tracks"].values()
                        for frame, box in track["visible"].items()))


def bootstrapped_median_difference(rows: list[dict], left: str, right: str,
                                   rng: np.random.Generator) -> dict:
    eligible = [row for row in rows if np.isfinite(float(row[left]))
                and np.isfinite(float(row[right]))]
    blocks = {}
    for row in eligible:
        blocks.setdefault(int(row["frame"]) // 20, []).append(float(row[left]) - float(row[right]))
    values = [value for chunk in blocks.values() for value in chunk]
    chunks = list(blocks.values())
    draws = []
    for _ in range(2000):
        sample = rng.integers(0, len(chunks), len(chunks))
        draws.append(float(np.median([value for index in sample for value in chunks[index]])))
    return {"valid_pairs": len(values), "median_left_minus_right_px": float(np.median(values)),
            "fraction_right_better": float(np.mean(np.asarray(values) > 0)),
            "block_bootstrap_95ci_px": np.quantile(draws, [0.025, 0.975]).tolist()}


def f1(counts: np.ndarray) -> float:
    tp, fp, fn = counts.tolist()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def association_difference(rows: list[dict], group: str, challenger: str,
                           reference: str, rng: np.random.Generator) -> dict:
    by_method = {method: {} for method in (reference, challenger)}
    for row in rows:
        method = row["method"]
        if method not in by_method:
            continue
        block = int(row["frame"]) // 20
        counts = np.asarray([int(row[key]) for key in ("tp", "fp", "fn")])
        by_method[method][block] = by_method[method].get(block, np.zeros(3, int)) + counts
    blocks = sorted(set(by_method[reference]) & set(by_method[challenger]))
    reference_counts = np.sum([by_method[reference][block] for block in blocks], axis=0)
    challenger_counts = np.sum([by_method[challenger][block] for block in blocks], axis=0)
    draws = []
    for _ in range(2000):
        sample = rng.choice(blocks, len(blocks), replace=True)
        reference_sample = np.sum([by_method[reference][int(block)] for block in sample], axis=0)
        challenger_sample = np.sum([by_method[challenger][int(block)] for block in sample], axis=0)
        draws.append(f1(challenger_sample) - f1(reference_sample))
    return {"group": group, "blocks": len(blocks),
            "challenger": challenger, "reference": reference,
            "challenger_f1": f1(challenger_counts), "reference_f1": f1(reference_counts),
            "f1_difference": f1(challenger_counts) - f1(reference_counts),
            "block_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    invariance = {}
    for input_name in INPUTS:
        baseline = signature(parse_xml(root / input_name / "none.xml"))
        invariance[input_name] = {"detection_count": len(baseline), "identical_boxes": {}}
        for method in METHODS:
            if not (root / input_name / f"{method}.xml").is_file():
                continue
            same = signature(parse_xml(root / input_name / f"{method}.xml")) == baseline
            invariance[input_name]["identical_boxes"][method] = same
            if not same:
                raise RuntimeError(f"Input boxes differ: {input_name}/{method}")

    rng = np.random.default_rng(20260924)
    geometry = csv_rows(root / "frame_geometry.csv")
    valid_geometry = [row for row in geometry if int(row["pose_valid"]) == 1
                      and int(row["background_features"]) >= 20]
    def geometry_comparison(reference: str, challenger: str) -> dict:
        return {
            label: bootstrapped_median_difference(
                valid_geometry, f"{reference}_{suffix}", f"{challenger}_{suffix}", rng
            )
            for label, suffix in (
                ("median_all", "median_px"), ("p90_all", "p90_px"),
                ("median_center", "center_median_px"),
                ("median_edge", "edge_median_px"),
            )
        }

    pairwise = csv_rows(root / "pairwise_per_frame.csv")
    comparisons = [("aria_vs_rgb", "aria_fisheye", "rgb_flow")]
    if (root / INPUTS[0] / "aria_plus_rgb_residual.xml").is_file():
        comparisons.extend((
            ("hybrid_vs_rgb", "aria_plus_rgb_residual", "rgb_flow"),
            ("hybrid_vs_aria", "aria_plus_rgb_residual", "aria_fisheye"),
            ("inverse_hybrid_vs_rgb", "inverse_plus_rgb_residual", "rgb_flow"),
        ))
    association = {name: {} for name, _, _ in comparisons}
    for input_name in INPUTS:
        selected = [row for row in pairwise if row["input"] == input_name]
        base = [row for row in selected if row["method"] == "none"]
        rotation = np.asarray([float(row["rotation_deg"]) for row in base])
        translation = np.asarray([float(row["translation_m"]) for row in base])
        scale = np.asarray([float(row["median_box_scale_px"]) for row in base])
        rot_low, rot_high = np.quantile(rotation, [1/3, 2/3])
        trans_low, trans_high = np.quantile(translation, [1/3, 2/3])
        scale_median = np.nanmedian(scale)
        predicates = {
            "all": lambda row: True,
            "high_rotation": lambda row: float(row["rotation_deg"]) >= rot_high,
            "low_rotation": lambda row: float(row["rotation_deg"]) <= rot_low,
            "high_translation": lambda row: float(row["translation_m"]) >= trans_high,
            "low_translation": lambda row: float(row["translation_m"]) <= trans_low,
            "small_boxes": lambda row: float(row["median_box_scale_px"]) < scale_median,
            "large_boxes": lambda row: float(row["median_box_scale_px"]) >= scale_median,
        }
        for name, challenger, reference in comparisons:
            association[name][input_name] = {
                group: association_difference(
                    [row for row in selected if predicate(row)],
                    group, challenger, reference, rng
                )
                for group, predicate in predicates.items()
            }

    result = {"coordinate_invariance": invariance,
              "geometry_comparisons": {
                  name: geometry_comparison(reference, challenger)
                  for name, challenger, reference in comparisons
              },
              "association_comparisons": association,
              "block_size_frames": 20, "bootstrap_draws": 2000,
              "note": "Positive differences mean challenger beats reference. Bootstrap intervals describe this one clip, not recording-level generalization."}
    path = root / "comparison_checks.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
