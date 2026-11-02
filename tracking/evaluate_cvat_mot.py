"""Evaluate CVAT video tracks against a corrected CVAT XML reference.

The evaluator expands CVAT interpolation before matching. It reports HOTA,
DetA and AssA through TrackEval, plus IDF1, ID switches, fragmentations and
detection errors through motmetrics. Prediction track IDs do not need to match
the numeric IDs used by the reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import motmetrics as mm
import numpy as np
import trackeval

from audit_cvat_video_annotations import parse_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=XML",
        help="Repeat for each prediction to evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def frame_index(dataset: dict) -> dict[int, list[tuple[int, list[float]]]]:
    frames = {frame: [] for frame in range(dataset["size"])}
    for track_id, track in dataset["tracks"].items():
        if track["label"] not in {"", "person"}:
            continue
        for frame, box in track["visible"].items():
            if 0 <= frame < dataset["size"]:
                frames[frame].append((track_id, box))
    return frames


def xyxy_to_xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def iou_matrix(left: list[list[float]], right: list[list[float]]) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=np.float64)
    for i, a in enumerate(left):
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        for j, b in enumerate(right):
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            result[i, j] = intersection / max(area_a + area_b - intersection, 1e-9)
    return result


def evaluate_one(gt: dict, prediction: dict, iou_threshold: float) -> dict:
    if gt["size"] != prediction["size"]:
        raise ValueError(f"Frame count mismatch: {gt['size']} != {prediction['size']}")

    gt_frames = frame_index(gt)
    pred_frames = frame_index(prediction)
    gt_ids_all = sorted({item[0] for rows in gt_frames.values() for item in rows})
    pred_ids_all = sorted({item[0] for rows in pred_frames.values() for item in rows})
    gt_remap = {track_id: index for index, track_id in enumerate(gt_ids_all)}
    pred_remap = {track_id: index for index, track_id in enumerate(pred_ids_all)}

    accumulator = mm.MOTAccumulator(auto_id=True)
    hota_data = {
        "num_timesteps": gt["size"],
        "num_gt_ids": len(gt_ids_all),
        "num_tracker_ids": len(pred_ids_all),
        "num_gt_dets": 0,
        "num_tracker_dets": 0,
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }

    for frame in range(gt["size"]):
        gt_rows = gt_frames[frame]
        pred_rows = pred_frames[frame]
        gt_ids = [row[0] for row in gt_rows]
        pred_ids = [row[0] for row in pred_rows]
        gt_boxes = [row[1] for row in gt_rows]
        pred_boxes = [row[1] for row in pred_rows]
        similarities = iou_matrix(gt_boxes, pred_boxes)

        distances = 1.0 - similarities
        distances[similarities < iou_threshold] = np.nan
        accumulator.update(gt_ids, pred_ids, distances)

        hota_data["gt_ids"].append(np.asarray([gt_remap[value] for value in gt_ids], dtype=int))
        hota_data["tracker_ids"].append(np.asarray([pred_remap[value] for value in pred_ids], dtype=int))
        hota_data["similarity_scores"].append(similarities)
        hota_data["num_gt_dets"] += len(gt_ids)
        hota_data["num_tracker_dets"] += len(pred_ids)

    mh = mm.metrics.create()
    requested = [
        "idf1",
        "idp",
        "idr",
        "mota",
        "motp",
        "precision",
        "recall",
        "num_switches",
        "num_fragmentations",
        "num_false_positives",
        "num_misses",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_unique_objects",
        "num_objects",
    ]
    mot = mh.compute(accumulator, metrics=requested, name="sequence").loc["sequence"]
    hota = trackeval.metrics.HOTA().eval_sequence(hota_data)

    result = {key: float(mot[key]) for key in requested}
    result.update(
        {
            "HOTA": float(np.mean(hota["HOTA"])),
            "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])),
            "LocA": float(np.mean(hota["LocA"])),
            "gt_tracks": len(gt_ids_all),
            "prediction_tracks": len(pred_ids_all),
            "gt_boxes": hota_data["num_gt_dets"],
            "prediction_boxes": hota_data["num_tracker_dets"],
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt = parse_xml(args.ground_truth)
    rows = []
    for item in args.prediction:
        if "=" not in item:
            raise ValueError(f"Prediction must be NAME=XML: {item}")
        name, raw_path = item.split("=", 1)
        row = evaluate_one(gt, parse_xml(Path(raw_path)), args.iou_threshold)
        row = {"name": name, "path": str(Path(raw_path).resolve()), **row}
        rows.append(row)

    rows.sort(key=lambda row: (row["idf1"], row["HOTA"], -row["num_switches"]), reverse=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "ground_truth": str(args.ground_truth.resolve()),
                "iou_threshold": args.iou_threshold,
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CVAT MOT comparison",
        "",
        f"Ground truth: `{args.ground_truth.resolve()}`",
        f"IoU matching threshold: {args.iou_threshold:.2f}",
        "",
        "| Candidate | HOTA | AssA | IDF1 | IDSW | Frag | Recall | Precision | Pred tracks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['HOTA']:.3f} | {row['AssA']:.3f} | {row['idf1']:.3f} | "
            f"{int(row['num_switches'])} | {int(row['num_fragmentations'])} | {row['recall']:.3f} | "
            f"{row['precision']:.3f} | {int(row['prediction_tracks'])} |"
        )
    lines.extend(
        [
            "",
            "HOTA/AssA measure overall detection-association quality. IDF1, IDSW and fragmentations are the primary identity-continuity gate.",
            "This benchmark is a pipeline-selection diagnostic on one corrected recording, not a cross-recording scientific claim.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
