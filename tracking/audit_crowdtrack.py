"""Audit a CrowdTrack release for structural and annotation quality.

The script is read-only unless --json-out is supplied. It accepts the extracted
CrowdTrack root containing train/train and test/test.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"_(\d+)\.[^.]+$")


def frame_number(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else None


def find_annotation(sequence: Path) -> Path | None:
    gt_dir = sequence / "gt"
    candidates = sorted(p for p in gt_dir.glob("*.txt") if p.stat().st_size > 0)
    if not candidates:
        return None
    preferred = gt_dir / "gt.txt"
    return preferred if preferred in candidates else candidates[0]


def audit_sequence(split: str, sequence: Path) -> dict:
    images = sorted(
        p for p in (sequence / "imgs").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    numbered = [(p, frame_number(p)) for p in images]
    image_frames = [n for _, n in numbered if n is not None]
    image_suffixes = Counter(p.suffix.lower() for p in images)

    width = height = None
    if images:
        with Image.open(images[0]) as image:
            width, height = image.size

    annotation = find_annotation(sequence)
    malformed = 0
    nonpositive_boxes = 0
    reversed_boxes = 0
    out_of_bounds = 0
    exact_duplicate_keys = 0
    conflicting_duplicate_keys = 0
    rows = 0
    frame_ids: set[int] = set()
    track_frames: dict[int, list[int]] = defaultdict(list)
    seen_keys: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    if annotation is not None:
        with annotation.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                fields = line.strip().replace(",", " ").split()
                if not fields:
                    continue
                if len(fields) < 6:
                    malformed += 1
                    continue
                try:
                    frame = int(float(fields[0]))
                    track_id = int(float(fields[1]))
                    x, y, w, h = map(float, fields[2:6])
                except ValueError:
                    malformed += 1
                    continue
                rows += 1
                frame_ids.add(frame)
                track_frames[track_id].append(frame)
                key = (frame, track_id)
                values = (x, y, w, h)
                if key in seen_keys:
                    if seen_keys[key] == values:
                        exact_duplicate_keys += 1
                    else:
                        conflicting_duplicate_keys += 1
                else:
                    seen_keys[key] = values
                if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    nonpositive_boxes += 1
                if w < 0 or h < 0:
                    reversed_boxes += 1
                if width is not None and height is not None:
                    left, right = sorted((x, x + w))
                    top, bottom = sorted((y, y + h))
                    if left < 0 or top < 0 or right > width + 1e-6 or bottom > height + 1e-6:
                        out_of_bounds += 1

    gap_events = 0
    gap_frames = 0
    for frames in track_frames.values():
        ordered = sorted(set(frames))
        for left, right in zip(ordered, ordered[1:]):
            if right - left > 1:
                gap_events += 1
                gap_frames += right - left - 1

    image_missing_numbers = 0
    if image_frames:
        image_missing_numbers = len(set(range(min(image_frames), max(image_frames) + 1)) - set(image_frames))

    annotation_frames_without_image = len(frame_ids - set(image_frames))
    image_frames_without_annotation = len(set(image_frames) - frame_ids)

    return {
        "split": split,
        "sequence": sequence.name,
        "annotation_file": str(annotation) if annotation else None,
        "images": len(images),
        "image_suffixes": dict(image_suffixes),
        "width": width,
        "height": height,
        "image_frame_min": min(image_frames) if image_frames else None,
        "image_frame_max": max(image_frames) if image_frames else None,
        "image_missing_numbers": image_missing_numbers,
        "annotation_rows": rows,
        "annotation_frames": len(frame_ids),
        "tracks": len(track_frames),
        "malformed_rows": malformed,
        "exact_duplicate_frame_track_keys": exact_duplicate_keys,
        "conflicting_duplicate_frame_track_keys": conflicting_duplicate_keys,
        "nonpositive_boxes": nonpositive_boxes,
        "reversed_boxes": reversed_boxes,
        "out_of_bounds_boxes_after_normalization": out_of_bounds,
        "annotation_frames_without_image": annotation_frames_without_image,
        "image_frames_without_annotation": image_frames_without_annotation,
        "track_gap_events": gap_events,
        "track_gap_frames": gap_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sequence_sets = [
        ("train", args.root / "train" / "train"),
        ("test", args.root / "test" / "test"),
    ]
    sequences = []
    for split, split_root in sequence_sets:
        if split_root.is_dir():
            sequences.extend(audit_sequence(split, p) for p in sorted(split_root.iterdir()) if p.is_dir())

    totals = {
        "sequences": len(sequences),
        "train_sequences": sum(s["split"] == "train" for s in sequences),
        "test_sequences": sum(s["split"] == "test" for s in sequences),
        "images": sum(s["images"] for s in sequences),
        "annotation_rows": sum(s["annotation_rows"] for s in sequences),
        "tracks_sum_per_sequence": sum(s["tracks"] for s in sequences),
        "malformed_rows": sum(s["malformed_rows"] for s in sequences),
        "exact_duplicate_frame_track_keys": sum(s["exact_duplicate_frame_track_keys"] for s in sequences),
        "conflicting_duplicate_frame_track_keys": sum(
            s["conflicting_duplicate_frame_track_keys"] for s in sequences
        ),
        "nonpositive_boxes": sum(s["nonpositive_boxes"] for s in sequences),
        "reversed_boxes": sum(s["reversed_boxes"] for s in sequences),
        "out_of_bounds_boxes_after_normalization": sum(
            s["out_of_bounds_boxes_after_normalization"] for s in sequences
        ),
        "image_missing_numbers": sum(s["image_missing_numbers"] for s in sequences),
        "annotation_frames_without_image": sum(s["annotation_frames_without_image"] for s in sequences),
        "image_frames_without_annotation": sum(s["image_frames_without_annotation"] for s in sequences),
        "track_gap_events": sum(s["track_gap_events"] for s in sequences),
        "track_gap_frames": sum(s["track_gap_frames"] for s in sequences),
    }
    result = {"root": str(args.root), "totals": totals, "sequences": sequences}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
