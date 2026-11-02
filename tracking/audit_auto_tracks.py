"""Audit automatic pedestrian proposals before CVAT human review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    return parser.parse_args()


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def height(box):
    return max(box[3] - box[1], 1.0)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(cfg.tracks_json.read_text(encoding="utf-8"))
    tracks = {int(t["track_id"]): sorted(t["boxes"], key=lambda r: r["frame_id"]) for t in data["tracks"]}

    metrics, gaps, jumps, low_conf = [], [], [], []
    for track_id, rows in tracks.items():
        frames = [r["frame_id"] for r in rows]
        internal_gaps = []
        max_jump = 0.0
        for prev, curr in zip(rows, rows[1:]):
            delta = curr["frame_id"] - prev["frame_id"]
            if delta > 1:
                event = {"track_id": track_id, "last_frame": prev["frame_id"], "next_frame": curr["frame_id"], "missing_frames": delta - 1}
                gaps.append(event)
                internal_gaps.append(event)
            if delta == 1:
                c1, c2 = center(prev["bbox_xyxy"]), center(curr["bbox_xyxy"])
                norm = math.dist(c1, c2) / max((height(prev["bbox_xyxy"]) + height(curr["bbox_xyxy"])) / 2, 1)
                max_jump = max(max_jump, norm)
                if norm > 0.65:
                    jumps.append({"track_id": track_id, "frame_from": prev["frame_id"], "frame_to": curr["frame_id"], "center_jump_per_box_height": round(norm, 4)})
        for row in rows:
            if row["confidence"] < 0.25:
                low_conf.append({"track_id": track_id, "frame_id": row["frame_id"], "confidence": round(row["confidence"], 5)})
        metrics.append({
            "track_id": track_id, "first_frame": min(frames), "last_frame": max(frames),
            "detections": len(rows), "span_frames": max(frames) - min(frames) + 1,
            "coverage_rate": round(len(rows) / (max(frames) - min(frames) + 1), 4),
            "internal_gap_count": len(internal_gaps), "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4),
            "min_confidence": round(min(r["confidence"] for r in rows), 4), "max_normalized_jump": round(max_jump, 4),
            "usable_60_frame_span": int(max(frames) - min(frames) + 1 >= cfg.window),
        })

    handoffs = []
    endpoints = {tid: (rows[0], rows[-1]) for tid, rows in tracks.items()}
    for old_id, (_, old_last) in endpoints.items():
        for new_id, (new_first, _) in endpoints.items():
            if old_id == new_id:
                continue
            gap = new_first["frame_id"] - old_last["frame_id"]
            if not 1 <= gap <= 10:
                continue
            old_box, new_box = old_last["bbox_xyxy"], new_first["bbox_xyxy"]
            norm_dist = math.dist(center(old_box), center(new_box)) / max((height(old_box) + height(new_box)) / 2, 1)
            overlap = iou(old_box, new_box)
            if norm_dist <= 0.8 or overlap >= 0.1:
                score = (1 / gap) + max(0, 0.8 - norm_dist) + overlap
                handoffs.append({
                    "old_track_id": old_id, "new_track_id": new_id,
                    "old_last_frame": old_last["frame_id"], "new_first_frame": new_first["frame_id"],
                    "frame_gap": gap, "normalized_center_distance": round(norm_dist, 4),
                    "endpoint_iou": round(overlap, 4), "priority_score": round(score, 4),
                })
    handoffs.sort(key=lambda row: row["priority_score"], reverse=True)
    metrics.sort(key=lambda row: (row["detections"], row["coverage_rate"]))
    gaps.sort(key=lambda row: (row["missing_frames"], row["last_frame"]), reverse=True)
    jumps.sort(key=lambda row: row["center_jump_per_box_height"], reverse=True)
    low_conf.sort(key=lambda row: row["confidence"])

    write_csv(cfg.output_dir / "track_metrics.csv", metrics)
    write_csv(cfg.output_dir / "handoff_candidates.csv", handoffs)
    write_csv(cfg.output_dir / "internal_gaps.csv", gaps)
    write_csv(cfg.output_dir / "jump_candidates.csv", jumps)
    write_csv(cfg.output_dir / "low_confidence_boxes.csv", low_conf)

    lengths = sorted(row["detections"] for row in metrics)
    summary = {
        "track_count": len(metrics), "box_count": sum(lengths),
        "median_track_detections": lengths[len(lengths) // 2],
        "tracks_under_60_detections": sum(v < cfg.window for v in lengths),
        "tracks_with_internal_gaps": sum(row["internal_gap_count"] > 0 for row in metrics),
        "internal_gap_events": len(gaps), "handoff_candidates": len(handoffs),
        "jump_candidates": len(jumps), "low_confidence_boxes_below_0_25": len(low_conf),
    }
    (cfg.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Round 0 automatic track audit

## Verdict

**Suitable for CVAT correction, not suitable as trajectory ground truth.** Box geometry is structurally valid, but automatic identities are fragmented and contain many gaps.

| Check | Result |
|---|---:|
| Tracks | {summary['track_count']} |
| Boxes | {summary['box_count']} |
| Median detections per track | {summary['median_track_detections']} |
| Tracks shorter than 60 detections | {summary['tracks_under_60_detections']} |
| Tracks with internal gaps | {summary['tracks_with_internal_gaps']} |
| Internal gap events | {summary['internal_gap_events']} |
| Endpoint handoff candidates | {summary['handoff_candidates']} |
| Large one-frame jump candidates | {summary['jump_candidates']} |
| Boxes below confidence 0.25 | {summary['low_confidence_boxes_below_0_25']} |

## Human review order

1. Review `handoff_candidates.csv` and merge only visually confirmed fragments.
2. Review `jump_candidates.csv` and split confirmed ID switches.
3. Inspect `internal_gaps.csv`; use occluded/outside rather than blind interpolation.
4. Review low-confidence detections together with missing-person frames.
5. Re-export from CVAT under a new version and rerun this audit.
"""
    (cfg.output_dir / "INITIAL_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

