"""Conservatively fuse two CVAT track proposals and build a review queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from audit_cvat_video_annotations import contiguous_runs, iou, parse_xml
from stitch_cvat_tracklets import write_cvat


EVENT_TYPES = (
    "MISSED_DETECTION",
    "DUPLICATE_TRACK",
    "POSSIBLE_RELINK",
    "ID_SWITCH",
    "TRACK_FRAGMENT",
    "BOX_JUMP",
    "TRACKER_DISAGREEMENT",
    "LONG_OCCLUSION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--deepoc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--duplicate-iou", type=float, default=0.75)
    parser.add_argument("--auto-merge-gap", type=int, default=20)
    parser.add_argument("--long-occlusion-gap", type=int, default=20)
    parser.add_argument("--max-relink-gap", type=int, default=90)
    parser.add_argument("--jump-threshold", type=float, default=1.0)
    parser.add_argument("--clip-padding", type=int, default=20)
    parser.add_argument("--max-review-clips", type=int, default=50)
    parser.add_argument("--max-checklist-events", type=int, default=100)
    parser.add_argument(
        "--apply-consensus-merges",
        action="store_true",
        help="Apply proposed cross-tracker relinks. Off by default so fusion cannot silently degrade the validated tracker.",
    )
    return parser.parse_args()


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def height(box):
    return max(1.0, box[3] - box[1])


def add_event(events: list[dict], event_type: str, start: int, end: int, score: float, **details) -> None:
    events.append(
        {
            "event_id": len(events) + 1,
            "event_type": event_type,
            "start_frame": int(start),
            "end_frame": int(end),
            "priority": round(float(score), 4),
            **details,
        }
    )


def frame_index(dataset: dict) -> dict[int, list[tuple[int, list[float]]]]:
    result = defaultdict(list)
    for track_id, track in dataset["tracks"].items():
        for frame, box in track["visible"].items():
            result[frame].append((track_id, box))
    return result


def pair_support(current: dict, deepoc: dict, threshold: float):
    current_frames, deep_frames = frame_index(current), frame_index(deepoc)
    supports = defaultdict(list)
    deep_unmatched = defaultdict(list)
    current_unmatched = defaultdict(list)
    for frame in range(min(current["size"], deepoc["size"])):
        current_rows = current_frames.get(frame, [])
        deep_rows = deep_frames.get(frame, [])
        current_best = {track_id: 0.0 for track_id, _ in current_rows}
        deep_best = {track_id: 0.0 for track_id, _ in deep_rows}
        for current_id, current_box in current_rows:
            for deep_id, deep_box in deep_rows:
                overlap = iou(current_box, deep_box)
                current_best[current_id] = max(current_best[current_id], overlap)
                deep_best[deep_id] = max(deep_best[deep_id], overlap)
                if overlap >= threshold:
                    supports[(current_id, deep_id)].append((frame, overlap))
        for track_id, best in current_best.items():
            if best < threshold:
                current_unmatched[track_id].append(frame)
        for track_id, best in deep_best.items():
            if best < threshold:
                deep_unmatched[track_id].append(frame)
    return supports, current_unmatched, deep_unmatched


def best_maps(supports):
    current_candidates = defaultdict(list)
    deep_candidates = defaultdict(list)
    for (current_id, deep_id), rows in supports.items():
        mean_iou = sum(row[1] for row in rows) / len(rows)
        item = (len(rows), mean_iou, current_id, deep_id)
        current_candidates[current_id].append(item)
        deep_candidates[deep_id].append(item)
    current_best = {
        current_id: max(rows) for current_id, rows in current_candidates.items()
    }
    deep_best = {deep_id: max(rows) for deep_id, rows in deep_candidates.items()}
    return current_candidates, deep_candidates, current_best, deep_best


def build_events(current: dict, deepoc: dict, args: argparse.Namespace):
    supports, current_unmatched, deep_unmatched = pair_support(current, deepoc, args.match_iou)
    current_candidates, deep_candidates, current_best, _ = best_maps(supports)
    events: list[dict] = []

    for deep_id, frames in deep_unmatched.items():
        for start, end in contiguous_runs(sorted(frames)):
            if end - start + 1 >= 2:
                add_event(
                    events,
                    "MISSED_DETECTION",
                    start,
                    end,
                    min(1.0, (end - start + 1) / 10),
                    current_track_ids=[],
                    deepoc_track_ids=[deep_id],
                    reason="Deep OC-SORT has a track with no overlapping current proposal.",
                )

    for current_id, frames in current_unmatched.items():
        for start, end in contiguous_runs(sorted(frames)):
            if end - start + 1 >= 3:
                add_event(
                    events,
                    "TRACKER_DISAGREEMENT",
                    start,
                    end,
                    min(1.0, (end - start + 1) / 15),
                    current_track_ids=[current_id],
                    deepoc_track_ids=[],
                    reason="Current high-recall tracker has a track unsupported by Deep OC-SORT.",
                )

    current_frames = frame_index(current)
    duplicate_frames = defaultdict(list)
    for frame, rows in current_frames.items():
        for index, (left_id, left_box) in enumerate(rows):
            for right_id, right_box in rows[index + 1 :]:
                overlap = iou(left_box, right_box)
                if overlap >= args.duplicate_iou:
                    duplicate_frames[tuple(sorted((left_id, right_id)))].append((frame, overlap))
    for (left_id, right_id), rows in duplicate_frames.items():
        by_frame = {row[0]: row[1] for row in rows}
        for start, end in contiguous_runs(sorted(by_frame)):
            peak = max(by_frame[frame] for frame in range(start, end + 1))
            add_event(
                events,
                "DUPLICATE_TRACK",
                start,
                end,
                peak,
                current_track_ids=[left_id, right_id],
                deepoc_track_ids=[],
                reason="Two current tracks overlap strongly in the same frames.",
            )

    for current_id, candidates in current_candidates.items():
        strong = sorted((item for item in candidates if item[0] >= 3), reverse=True)
        if len(strong) >= 2:
            deep_ids = [item[3] for item in strong]
            frames = sorted(current["tracks"][current_id]["visible"])
            add_event(
                events,
                "ID_SWITCH",
                frames[0],
                frames[-1],
                min(1.0, strong[1][0] / max(strong[0][0], 1)),
                current_track_ids=[current_id],
                deepoc_track_ids=deep_ids,
                reason="One current track is supported by multiple Deep OC-SORT identities.",
            )

    merge_pairs = []
    for deep_id, candidates in deep_candidates.items():
        strong_current = sorted(
            {item[2] for item in candidates if item[0] >= 3},
            key=lambda track_id: min(current["tracks"][track_id]["visible"]),
        )
        for left_id, right_id in zip(strong_current, strong_current[1:]):
            left_frames = sorted(current["tracks"][left_id]["visible"])
            right_frames = sorted(current["tracks"][right_id]["visible"])
            if left_frames[-1] >= right_frames[0]:
                continue
            gap = right_frames[0] - left_frames[-1] - 1
            if gap <= args.max_relink_gap:
                kind = "LONG_OCCLUSION" if gap > args.long_occlusion_gap else "POSSIBLE_RELINK"
                add_event(
                    events,
                    kind,
                    left_frames[-1],
                    right_frames[0],
                    1.0 - min(gap, args.max_relink_gap) / max(args.max_relink_gap, 1),
                    current_track_ids=[left_id, right_id],
                    deepoc_track_ids=[deep_id],
                    gap_frames=gap,
                    reason="Deep OC-SORT connects two non-overlapping current tracklets.",
                )
                add_event(
                    events,
                    "TRACK_FRAGMENT",
                    left_frames[-1],
                    right_frames[0],
                    0.9 if gap <= args.auto_merge_gap else 0.6,
                    current_track_ids=[left_id, right_id],
                    deepoc_track_ids=[deep_id],
                    gap_frames=gap,
                    reason="Multiple current tracklets map to one Deep OC-SORT identity.",
                )
                if gap <= args.auto_merge_gap:
                    merge_pairs.append((left_id, right_id))

    jump_candidates = defaultdict(list)
    for track_id, track in current["tracks"].items():
        frames = sorted(track["visible"])
        for previous, frame in zip(frames, frames[1:]):
            if frame != previous + 1:
                continue
            left, right = track["visible"][previous], track["visible"][frame]
            jump = math.dist(center(left), center(right)) / (0.5 * (height(left) + height(right)))
            if jump > args.jump_threshold:
                jump_candidates[track_id].append((previous, frame, jump))

    # Egocentric camera motion can trigger many adjacent one-frame jumps. Group
    # nearby candidates from the same identity into one review event.
    for track_id, candidates in jump_candidates.items():
        groups = []
        current_group = []
        for candidate in candidates:
            if current_group and candidate[0] - current_group[-1][1] > 5:
                groups.append(current_group)
                current_group = []
            current_group.append(candidate)
        if current_group:
            groups.append(current_group)
        for group in groups:
            peak = max(item[2] for item in group)
            start, end = group[0][0], group[-1][1]
            if peak > args.jump_threshold:
                add_event(
                    events,
                    "BOX_JUMP",
                    start,
                    end,
                    min(1.0, peak / 2.0),
                    current_track_ids=[track_id],
                    deepoc_track_ids=[],
                    normalized_jump=round(peak, 4),
                    candidate_frames=len(group),
                    reason="Grouped one-frame displacements exceed the normalized jump gate.",
                )

    events.sort(key=lambda row: (-row["priority"], row["start_frame"], row["event_type"]))
    for event_id, event in enumerate(events, start=1):
        event["event_id"] = event_id
    return events, merge_pairs, current_best


def make_chains(dataset: dict, merge_pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = {track_id: track_id for track_id in dataset["tracks"] if dataset["tracks"][track_id]["visible"]}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in merge_pairs:
        union(left, right)
    groups = defaultdict(list)
    for track_id in parent:
        groups[find(track_id)].append(track_id)
    return [
        sorted(group, key=lambda track_id: min(dataset["tracks"][track_id]["visible"]))
        for group in groups.values()
    ]


def write_review_clips(video: Path, current: dict, deepoc: dict, events: list[dict], output: Path, args) -> int:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_frames, deep_frames = frame_index(current), frame_index(deepoc)
    count = 0
    for event in events[: args.max_review_clips]:
        start = max(0, event["start_frame"] - args.clip_padding)
        end = min(current["size"] - 1, event["end_frame"] + args.clip_padding)
        path = output / f"{event['event_id']:04d}_{event['event_type']}_{start}_{end}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_id in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            for track_id, box in current_frames.get(frame_id, []):
                x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"C{track_id}", (x1, max(25, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for track_id, box in deep_frames.get(frame_id, []):
                x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"D{track_id}", (x1, min(height - 5, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, f"#{event['event_id']} {event['event_type']} frame {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            writer.write(frame)
        writer.release()
        event["review_clip"] = str(path.resolve())
        count += 1
    capture.release()
    return count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current, deepoc = parse_xml(args.current), parse_xml(args.deepoc)
    if current["size"] != deepoc["size"]:
        raise ValueError(f"Frame count mismatch: {current['size']} != {deepoc['size']}")
    events, merge_pairs, current_best = build_events(current, deepoc, args)
    applied_merge_pairs = merge_pairs if args.apply_consensus_merges else []
    chains = make_chains(current, applied_merge_pairs)
    write_cvat(args.output_dir / "annotations.xml", current, chains)
    clip_count = write_review_clips(
        args.video, current, deepoc, events, args.output_dir / "review_clips", args
    )

    (args.output_dir / "conflicts.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    flat_rows = []
    for event in events[: args.max_checklist_events]:
        flat_rows.append(
            {
                **event,
                "current_track_ids": ";".join(map(str, event.get("current_track_ids", []))),
                "deepoc_track_ids": ";".join(map(str, event.get("deepoc_track_ids", []))),
            }
        )
    if flat_rows:
        with (args.output_dir / "conflicts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat_rows for key in row}))
            writer.writeheader()
            writer.writerows(flat_rows)
    else:
        (args.output_dir / "conflicts.csv").write_text("", encoding="utf-8")

    counts = {kind: sum(event["event_type"] == kind for event in events) for kind in EVENT_TYPES}
    lines = [
        "# Phase 1 conflict review queue",
        "",
        f"Fused CVAT proposal: `{(args.output_dir / 'annotations.xml').resolve()}`",
        f"Total events: **{len(events)}**; rendered review clips: **{clip_count}**.",
        "",
        "## Event counts",
        "",
        "| Event | Count |",
        "|---|---:|",
        *[f"| {kind} | {counts[kind]} |" for kind in EVENT_TYPES],
        "",
        "## Review order",
        "",
        "Red boxes/IDs are the current high-recall tracker (`C`); cyan boxes/IDs are Deep OC-SORT (`D`).",
        "Review rows in priority order. In CVAT, merge/split/delete/add boxes only after checking temporal context.",
        "",
        "| # | Priority | Type | Frames | Current IDs | DeepOC IDs |",
        "|---:|---:|---|---:|---|---|",
    ]
    for event in events:
        lines.append(
            f"| {event['event_id']} | {event['priority']:.3f} | {event['event_type']} | "
            f"{event['start_frame']}-{event['end_frame']} | "
            f"{','.join(map(str, event.get('current_track_ids', []))) or '—'} | "
            f"{','.join(map(str, event.get('deepoc_track_ids', []))) or '—'} |"
        )
    (args.output_dir / "REVIEW_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "current": str(args.current.resolve()),
        "deepoc": str(args.deepoc.resolve()),
        "event_types": list(EVENT_TYPES),
        "event_counts": counts,
        "events": len(events),
        "review_clips": clip_count,
        "proposed_merge_pairs": merge_pairs,
        "applied_merge_pairs": applied_merge_pairs,
        "current_to_deepoc_best": {
            str(current_id): {
                "support_frames": row[0],
                "mean_iou": round(row[1], 6),
                "deepoc_track_id": row[3],
            }
            for current_id, row in current_best.items()
        },
        "fused_tracks": len(chains),
        "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
    }
    (args.output_dir / "fusion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"events": len(events), "counts": counts, "fused_tracks": len(chains), "clips": clip_count}, indent=2))


if __name__ == "__main__":
    main()
