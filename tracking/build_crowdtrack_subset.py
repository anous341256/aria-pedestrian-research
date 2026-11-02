"""Build a read-only manifest for the Aria-relevant, structurally clean CrowdTrack subset.

The source release is never modified. The output manifest selects complete track
IDs whose rows are unique, finite, positive-sized, inside the image, and backed
by an existing frame image.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"_(\d+)\.[^.]+$")

# Mid-frame visual review: moving pedestrian-height views in malls, metro
# stations, pedestrian streets, or public plazas.
DOMAIN_RELEVANT = {
    "track0001",
    "track0002",
    "track0003",
    "track0004",
    "track0005",
    "track0006",
    "track0007",
    "track0008",
    "track0009",
    "track0011",
    "track0012",
    "track0013",
    "track0014",
    "track0015",
    "track0016",
    "track0017",
    "track0031",
    "track0032",
    "track0033",
}

# Excluded at sequence level because structural defects are concentrated enough
# to make the raw sequence a poor no-repair training source.
SEQUENCE_EXCLUSIONS = {
    "track0001": "structural filtering would remove 4,594/15,096 annotation rows (30.4%)",
    "track0002": "56 reversed/nonpositive boxes and inconsistent test GT filename",
    "track0003": "2,248 nonpositive boxes and 3,704 normalized out-of-bounds boxes",
    "track0004": "annotation references 8 missing image frames and has 317 nonpositive boxes",
    "track0009": "structural filtering would remove 6,487/12,231 annotation rows (53.0%)",
    "track0013": "175 conflicting duplicate frame-ID keys",
    "track0014": "47 conflicting duplicate keys and 4,745 normalized out-of-bounds boxes",
}


def image_frame(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else None


def annotation_file(sequence: Path) -> Path:
    files = sorted(p for p in (sequence / "gt").glob("*.txt") if p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"No non-empty annotation file in {sequence}")
    preferred = sequence / "gt" / "gt.txt"
    return preferred if preferred in files else files[0]


def select_sequence(split: str, sequence: Path) -> dict:
    images = sorted(
        p for p in (sequence / "imgs").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    frame_to_image = {image_frame(p): p for p in images if image_frame(p) is not None}
    with Image.open(images[0]) as image:
        width, height = image.size

    ann = annotation_file(sequence)
    rows_by_track: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    key_values: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    bad_reasons: dict[int, set[str]] = defaultdict(set)

    with ann.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.strip().replace(",", " ").split()
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, w, h = map(float, fields[2:6])
            rows_by_track[track_id].append((frame, x, y, w, h))
            values = (x, y, w, h)
            key = (frame, track_id)

            if key in key_values and key_values[key] != values:
                bad_reasons[track_id].add("conflicting_duplicate_frame_id")
            else:
                key_values.setdefault(key, values)
            if not all(math.isfinite(v) for v in values):
                bad_reasons[track_id].add("nonfinite_coordinate")
            if w <= 0 or h <= 0:
                bad_reasons[track_id].add("nonpositive_box")
            left, right = sorted((x, x + w))
            top, bottom = sorted((y, y + h))
            if left < 0 or top < 0 or right > width or bottom > height:
                bad_reasons[track_id].add("out_of_bounds_box")
            if frame not in frame_to_image:
                bad_reasons[track_id].add("annotation_without_image")

    selected_ids = sorted(track_id for track_id in rows_by_track if not bad_reasons[track_id])
    excluded_ids = sorted(track_id for track_id in rows_by_track if bad_reasons[track_id])
    selected_rows = sum(len(rows_by_track[track_id]) for track_id in selected_ids)
    excluded_rows = sum(len(rows_by_track[track_id]) for track_id in excluded_ids)

    return {
        "split": split,
        "sequence": sequence.name,
        "sequence_path": str(sequence),
        "images_path": str(sequence / "imgs"),
        "annotation_path": str(ann),
        "width": width,
        "height": height,
        "image_frames": len(images),
        "selected_track_ids": selected_ids,
        "excluded_tracks": [
            {"track_id": track_id, "reasons": sorted(bad_reasons[track_id])}
            for track_id in excluded_ids
        ],
        "selected_tracks": len(selected_ids),
        "excluded_tracks_count": len(excluded_ids),
        "selected_annotation_rows": selected_rows,
        "excluded_annotation_rows": excluded_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_sequences: dict[str, tuple[str, Path]] = {}
    for split in ("train", "test"):
        split_root = args.root / split / split
        for sequence in split_root.iterdir():
            if sequence.is_dir():
                all_sequences[sequence.name] = (split, sequence)

    selected = []
    for name in sorted(DOMAIN_RELEVANT - set(SEQUENCE_EXCLUSIONS)):
        split, sequence = all_sequences[name]
        selected.append(select_sequence(split, sequence))

    totals = {
        "selected_sequences": len(selected),
        "train_sequences": sum(s["split"] == "train" for s in selected),
        "test_sequences": sum(s["split"] == "test" for s in selected),
        "image_frames": sum(s["image_frames"] for s in selected),
        "selected_tracks": sum(s["selected_tracks"] for s in selected),
        "excluded_tracks": sum(s["excluded_tracks_count"] for s in selected),
        "selected_annotation_rows": sum(s["selected_annotation_rows"] for s in selected),
        "excluded_annotation_rows": sum(s["excluded_annotation_rows"] for s in selected),
    }
    payload = {
        "schema_version": 1,
        "source_root": str(args.root),
        "selection_policy": {
            "domain": "moving pedestrian-height mall, metro, pedestrian-street, or plaza scenes",
            "track_rule": "keep a complete ID only when every row is unique, finite, positive-sized, in bounds, and has an image",
            "source_modified": False,
        },
        "sequence_exclusions": SEQUENCE_EXCLUSIONS,
        "totals": totals,
        "sequences": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
