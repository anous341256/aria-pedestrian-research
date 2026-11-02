"""Audit CVAT video-track XML exports and compare them with a baseline export.

The audit expands CVAT's linear interpolation so that checks reflect the boxes
shown in the editor, not only the stored keyframes. Candidate lists are review
queues: camera motion and genuine pedestrian overlap can trigger them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_xml", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-xml", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--overlap-iou", type=float, default=0.70)
    parser.add_argument("--jump-threshold", type=float, default=0.65)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_from_element(element):
    return [float(element.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]


def lerp_box(a, b, ratio):
    return [x + (y - x) * ratio for x, y in zip(a, b)]


def expand_track(keyframes, stop_frame):
    """Return visible frame -> box using CVAT interpolation semantics."""
    visible = {}
    for index, current in enumerate(keyframes):
        frame = current["frame"]
        if not current["outside"]:
            visible[frame] = current["box"]
        if index + 1 >= len(keyframes):
            if not current["outside"]:
                for target in range(frame + 1, stop_frame + 1):
                    visible[target] = current["box"]
            continue
        nxt = keyframes[index + 1]
        if current["outside"]:
            continue
        span = nxt["frame"] - frame
        if span <= 0:
            continue
        final_visible = nxt["frame"] - 1 if nxt["outside"] else nxt["frame"]
        for target in range(frame + 1, final_visible + 1):
            ratio = (target - frame) / span
            visible[target] = lerp_box(current["box"], nxt["box"], ratio)
    return visible


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    width_node = root.find("./meta/original_size/width")
    height_node = root.find("./meta/original_size/height")
    size_node = root.find("./meta/job/size") or root.find("./meta/task/size")
    stop_node = root.find("./meta/job/stop_frame")
    width = int(width_node.text) if width_node is not None else 2560
    height = int(height_node.text) if height_node is not None else 1920
    size = int(size_node.text) if size_node is not None else 600
    stop_frame = int(stop_node.text) if stop_node is not None else size - 1
    tracks = {}
    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        keyframes = []
        for box in track.findall("box"):
            keyframes.append({
                "frame": int(box.attrib["frame"]),
                "outside": box.attrib.get("outside", "0") == "1",
                "occluded": box.attrib.get("occluded", "0") == "1",
                "keyframe": box.attrib.get("keyframe", "1") == "1",
                "box": box_from_element(box),
            })
        keyframes.sort(key=lambda row: row["frame"])
        tracks[track_id] = {
            "label": track.attrib.get("label", ""),
            "source": track.attrib.get("source", "unknown"),
            "keyframes": keyframes,
            "visible": expand_track(keyframes, stop_frame),
        }
    return {"width": width, "height": height, "size": size, "stop_frame": stop_frame, "tracks": tracks}


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_height(box):
    return max(box[3] - box[1], 1.0)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(area(a) + area(b) - intersection, 1e-9)


def contiguous_runs(frames):
    if not frames:
        return []
    runs, start, previous = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    return runs


def summarize(dataset, window, overlap_threshold, jump_threshold):
    width, height, tracks = dataset["width"], dataset["height"], dataset["tracks"]
    track_metrics, invalid, out_of_bounds = [], [], []
    gaps, jumps, size_jumps, duplicate_keyframes = [], [], [], []
    frame_objects = defaultdict(list)
    source_track_counts = Counter()
    source_keyframe_counts = Counter()
    occluded_keyframes = 0
    outside_keyframes = 0

    for track_id, track in tracks.items():
        source_track_counts[track["source"]] += 1
        source_keyframe_counts[track["source"]] += len(track["keyframes"])
        seen_frames = Counter(row["frame"] for row in track["keyframes"])
        for frame, count in seen_frames.items():
            if count > 1:
                duplicate_keyframes.append({"track_id": track_id, "frame": frame, "count": count})
        for row in track["keyframes"]:
            occluded_keyframes += int(row["occluded"] and not row["outside"])
            outside_keyframes += int(row["outside"])

        frames = sorted(track["visible"])
        runs = contiguous_runs(frames)
        for previous, current in zip(runs, runs[1:]):
            gaps.append({
                "track_id": track_id,
                "last_visible_frame": previous[1],
                "next_visible_frame": current[0],
                "missing_frames": current[0] - previous[1] - 1,
            })

        max_jump = 0.0
        for frame in frames:
            box = track["visible"][frame]
            frame_objects[frame].append((track_id, box))
            if box[2] <= box[0] or box[3] <= box[1]:
                invalid.append({"track_id": track_id, "frame": frame, "box": box})
            if box[0] < -0.01 or box[1] < -0.01 or box[2] > width + 0.01 or box[3] > height + 0.01:
                out_of_bounds.append({"track_id": track_id, "frame": frame, "box": box})
            if frame - 1 not in track["visible"]:
                continue
            previous = track["visible"][frame - 1]
            norm = math.dist(center(previous), center(box)) / max((box_height(previous) + box_height(box)) / 2, 1)
            max_jump = max(max_jump, norm)
            if norm > jump_threshold:
                jumps.append({
                    "track_id": track_id,
                    "frame_from": frame - 1,
                    "frame_to": frame,
                    "center_jump_per_box_height": round(norm, 4),
                })
            ratio = max(area(previous), area(box)) / max(min(area(previous), area(box)), 1e-9)
            if ratio > 2.5:
                size_jumps.append({
                    "track_id": track_id,
                    "frame_from": frame - 1,
                    "frame_to": frame,
                    "area_ratio": round(ratio, 4),
                })

        track_metrics.append({
            "track_id": track_id,
            "source": track["source"],
            "keyframes": len(track["keyframes"]),
            "visible_frames": len(frames),
            "first_visible_frame": min(frames) if frames else "",
            "last_visible_frame": max(frames) if frames else "",
            "visible_runs": len(runs),
            "internal_gap_count": max(0, len(runs) - 1),
            "max_normalized_jump": round(max_jump, 4),
            "usable_60_frame_span": int(len(frames) >= window),
        })

    overlap_events = []
    for frame, objects in frame_objects.items():
        for index, (left_id, left_box) in enumerate(objects):
            for right_id, right_box in objects[index + 1:]:
                overlap = iou(left_box, right_box)
                if overlap >= overlap_threshold:
                    overlap_events.append({
                        "frame": frame,
                        "track_id_a": min(left_id, right_id),
                        "track_id_b": max(left_id, right_id),
                        "iou": round(overlap, 4),
                    })

    overlap_runs = []
    grouped = defaultdict(list)
    for row in overlap_events:
        grouped[(row["track_id_a"], row["track_id_b"])].append(row)
    for (left_id, right_id), rows in grouped.items():
        by_frame = {row["frame"]: row for row in rows}
        for start, end in contiguous_runs(sorted(by_frame)):
            run_rows = [by_frame[frame] for frame in range(start, end + 1)]
            overlap_runs.append({
                "track_id_a": left_id,
                "track_id_b": right_id,
                "start_frame": start,
                "end_frame": end,
                "frames": end - start + 1,
                "max_iou": max(row["iou"] for row in run_rows),
                "mean_iou": round(sum(row["iou"] for row in run_rows) / len(run_rows), 4),
            })

    jumps.sort(key=lambda row: row["center_jump_per_box_height"], reverse=True)
    size_jumps.sort(key=lambda row: row["area_ratio"], reverse=True)
    overlap_events.sort(key=lambda row: row["iou"], reverse=True)
    overlap_runs.sort(key=lambda row: (row["frames"], row["max_iou"]), reverse=True)
    gaps.sort(key=lambda row: row["missing_frames"], reverse=True)
    track_metrics.sort(key=lambda row: (row["visible_frames"], row["track_id"]))

    lengths = sorted(row["visible_frames"] for row in track_metrics)
    median_length = lengths[len(lengths) // 2] if lengths else 0
    empty_tracks = [
        {"track_id": row["track_id"], "source": row["source"], "stored_keyframes": row["keyframes"]}
        for row in track_metrics if row["visible_frames"] == 0
    ]
    summary = {
        "frame_count": dataset["size"],
        "track_count": len(tracks),
        "active_track_count": len(tracks) - len(empty_tracks),
        "empty_tracks": len(empty_tracks),
        "rendered_visible_boxes": sum(lengths),
        "stored_keyframes": sum(len(track["keyframes"]) for track in tracks.values()),
        "median_visible_frames_per_track": median_length,
        "tracks_under_60_visible_frames": sum(length < window for length in lengths),
        "tracks_with_internal_visibility_gaps": sum(row["internal_gap_count"] > 0 for row in track_metrics),
        "internal_visibility_gap_events": len(gaps),
        "high_iou_overlap_events": len(overlap_events),
        "high_iou_overlap_runs": len(overlap_runs),
        "large_one_frame_jump_candidates": len(jumps),
        "large_area_change_candidates": len(size_jumps),
        "invalid_boxes": len(invalid),
        "out_of_bounds_boxes": len(out_of_bounds),
        "duplicate_track_frame_keyframes": len(duplicate_keyframes),
        "occluded_visible_keyframes": occluded_keyframes,
        "outside_keyframes": outside_keyframes,
        "track_sources": dict(source_track_counts),
        "keyframe_sources": dict(source_keyframe_counts),
    }
    details = {
        "track_metrics": track_metrics,
        "gaps": gaps,
        "jumps": jumps,
        "size_jumps": size_jumps,
        "overlap_events": overlap_events,
        "overlap_runs": overlap_runs,
        "invalid": invalid,
        "out_of_bounds": out_of_bounds,
        "duplicate_keyframes": duplicate_keyframes,
        "empty_tracks": empty_tracks,
    }
    return summary, details


def delta(current, baseline, key):
    return current.get(key, 0) - baseline.get(key, 0)


def make_contact_sheet(video_path, dataset, details, output_path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    candidates = []
    for row in details["overlap_runs"][:8]:
        candidates.append((row["start_frame"], f"overlap {row['track_id_a']}/{row['track_id_b']} IoU {row['max_iou']:.2f}"))
    for row in details["size_jumps"][:8]:
        candidates.append((row["frame_to"], f"scale ID {row['track_id']} x{row['area_ratio']:.2f}"))
    for row in details["jumps"][:8]:
        candidates.append((row["frame_to"], f"jump ID {row['track_id']} {row['center_jump_per_box_height']:.2f}h"))
    for row in details["gaps"][:8]:
        candidates.append((row["next_visible_frame"], f"gap ID {row['track_id']} {row['missing_frames']}f"))
    deduped = []
    seen = set()
    for item in candidates:
        if item[0] not in seen:
            deduped.append(item)
            seen.add(item[0])
    candidates = deduped[:24]
    if not candidates:
        return False
    capture = cv2.VideoCapture(str(video_path))
    tiles = []
    frames_dir = output_path.parent / "qa_candidate_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, reason in candidates:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        for track_id, track in dataset["tracks"].items():
            if frame_index not in track["visible"]:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in track["visible"][frame_index]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, str(track_id), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"frame {frame_index}: {reason}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"), frame)
        scale = 640 / frame.shape[1]
        tiles.append(cv2.resize(frame, (640, int(frame.shape[0] * scale))))
    capture.release()
    if not tiles:
        return False
    tile_height = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        canvas = np.full((tile_height, 640, 3), 255, dtype=np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        padded.append(canvas)
    rows = []
    for index in range(0, len(padded), 3):
        chunk = padded[index:index + 3]
        while len(chunk) < 3:
            chunk.append(np.full_like(padded[0], 255))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    cv2.imwrite(str(output_path), sheet)
    return True


def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    current_dataset = parse_xml(cfg.annotations_xml)
    current, details = summarize(current_dataset, cfg.window, cfg.overlap_iou, cfg.jump_threshold)
    baseline = None
    if cfg.baseline_xml:
        baseline_dataset = parse_xml(cfg.baseline_xml)
        baseline, _ = summarize(baseline_dataset, cfg.window, cfg.overlap_iou, cfg.jump_threshold)

    write_csv(cfg.output_dir / "track_metrics.csv", details["track_metrics"])
    write_csv(cfg.output_dir / "internal_visibility_gaps.csv", details["gaps"])
    write_csv(cfg.output_dir / "jump_candidates.csv", details["jumps"])
    write_csv(cfg.output_dir / "area_change_candidates.csv", details["size_jumps"])
    write_csv(cfg.output_dir / "high_iou_overlap_events.csv", details["overlap_events"])
    write_csv(cfg.output_dir / "high_iou_overlap_runs.csv", details["overlap_runs"])
    write_csv(cfg.output_dir / "invalid_boxes.csv", details["invalid"])
    write_csv(cfg.output_dir / "out_of_bounds_boxes.csv", details["out_of_bounds"])
    write_csv(cfg.output_dir / "duplicate_track_frame_keyframes.csv", details["duplicate_keyframes"])
    write_csv(cfg.output_dir / "empty_tracks.csv", details["empty_tracks"])
    payload = {"current": current, "baseline": baseline}
    (cfg.output_dir / "qa_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    comparison_rows = ""
    if baseline:
        keys = [
            "track_count", "active_track_count", "empty_tracks", "rendered_visible_boxes", "median_visible_frames_per_track",
            "tracks_under_60_visible_frames", "tracks_with_internal_visibility_gaps",
            "internal_visibility_gap_events", "high_iou_overlap_events",
            "high_iou_overlap_runs", "large_one_frame_jump_candidates",
            "large_area_change_candidates", "invalid_boxes", "out_of_bounds_boxes",
        ]
        comparison_rows = "\n".join(
            f"| {key} | {baseline.get(key, 0)} | {current.get(key, 0)} | {delta(current, baseline, key):+d} |"
            for key in keys
        )

    verdict = "Structurally usable, with targeted visual review still required."
    if current["invalid_boxes"] or current["out_of_bounds_boxes"] or current["duplicate_track_frame_keyframes"]:
        verdict = "Not yet structurally clean; repair invalid or duplicate records before training."
    report = f"""# CVAT Round 1 annotation QA

## Verdict

**{verdict}** High-IoU overlaps and large jumps are candidates rather than confirmed errors because genuine occlusion and fast egocentric camera motion can trigger them.

## Current export

| Check | Result |
|---|---:|
| Frames | {current['frame_count']} |
| Tracks | {current['track_count']} |
| Tracks with at least one visible box | {current['active_track_count']} |
| Empty tracks (all keyframes outside) | {current['empty_tracks']} |
| Rendered visible boxes | {current['rendered_visible_boxes']} |
| Stored keyframes | {current['stored_keyframes']} |
| Median visible frames per track | {current['median_visible_frames_per_track']} |
| Tracks under 60 visible frames | {current['tracks_under_60_visible_frames']} |
| Tracks with internal visibility gaps | {current['tracks_with_internal_visibility_gaps']} |
| Internal visibility-gap events | {current['internal_visibility_gap_events']} |
| High-IoU overlap events | {current['high_iou_overlap_events']} |
| High-IoU overlap runs | {current['high_iou_overlap_runs']} |
| Large one-frame jump candidates | {current['large_one_frame_jump_candidates']} |
| Large area-change candidates | {current['large_area_change_candidates']} |
| Invalid boxes | {current['invalid_boxes']} |
| Out-of-bounds boxes | {current['out_of_bounds_boxes']} |
| Duplicate track/frame keyframes | {current['duplicate_track_frame_keyframes']} |
| Visible keyframes marked occluded | {current['occluded_visible_keyframes']} |

## Round 0 comparison

| Metric | Round 0 | Round 1 | Delta |
|---|---:|---:|---:|
{comparison_rows if comparison_rows else '| Baseline | unavailable | unavailable | — |'}

## Review order

1. Inspect `high_iou_overlap_runs.csv` for duplicate IDs. Long runs with very high IoU are the strongest duplicate-track signal.
2. Inspect the top of `jump_candidates.csv` for ID switches or misplaced boxes; fast camera motion can be a false positive.
3. Inspect `area_change_candidates.csv` for abrupt scale errors.
4. Review `internal_visibility_gaps.csv`; confirm that every gap represents a real disappearance/occlusion rather than a missed box.
5. Treat `source=manual` only as provenance, not as proof of correctness.

## Scope and assumptions

- Boxes are expanded using CVAT linear interpolation between stored keyframes.
- A high-overlap event uses IoU >= {cfg.overlap_iou:.2f}; overlapping pedestrians can legitimately exceed this threshold.
- A jump candidate uses center displacement > {cfg.jump_threshold:.2f} times mean box height in one frame.
- This audit cannot determine identity correctness without visual inspection.
"""
    (cfg.output_dir / "ROUND1_QA_REPORT.md").write_text(report, encoding="utf-8")

    if cfg.video:
        make_contact_sheet(cfg.video, current_dataset, details, cfg.output_dir / "qa_candidate_contact_sheet.jpg")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
