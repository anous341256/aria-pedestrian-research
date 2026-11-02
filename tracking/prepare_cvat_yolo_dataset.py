"""Convert a CVAT video-track export into a versioned YOLO detection dataset.

CVAT interpolation is expanded frame by frame.  Track identities are deliberately
not written to YOLO labels: YOLO learns person detection, while the tracker and
human review remain responsible for identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_cvat_video_annotations import parse_xml


def parse_range(value: str) -> range:
    """Parse a half-open START:STOP frame range."""
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("range must be START:STOP") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < STOP")
    return range(start, stop)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_row(box: list[float], width: int, height: int) -> str | None:
    x1 = min(max(box[0], 0.0), float(width))
    y1 = min(max(box[1], 0.0), float(height))
    x2 = min(max(box[2], 0.0), float(width))
    y2 = min(max(box[3], 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations_xml", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-range", type=parse_range, default=parse_range("0:480"))
    parser.add_argument("--val-range", type=parse_range, default=parse_range("520:600"))
    args = parser.parse_args()

    if not args.annotations_xml.is_file():
        raise FileNotFoundError(args.annotations_xml)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    import cv2
    import numpy as np

    dataset = parse_xml(args.annotations_xml)
    frame_count = dataset["size"]
    split_frames = {
        "train": set(args.train_range),
        "val": set(args.val_range),
    }
    if split_frames["train"] & split_frames["val"]:
        raise ValueError("train and val frame ranges overlap")
    for split, frames in split_frames.items():
        invalid = [frame for frame in frames if frame >= frame_count]
        if invalid:
            raise ValueError(f"{split} range exceeds annotation size {frame_count}")

    for split in split_frames:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    frame_boxes: dict[int, list[list[float]]] = {frame: [] for frame in range(frame_count)}
    for track in dataset["tracks"].values():
        if track["label"] != "person":
            continue
        for frame, box in track["visible"].items():
            frame_boxes.setdefault(frame, []).append(box)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (video_width, video_height) != (dataset["width"], dataset["height"]):
        raise ValueError(
            "video/XML dimensions differ: "
            f"video={video_width}x{video_height}, XML={dataset['width']}x{dataset['height']}"
        )

    stats = {
        split: {"frames": 0, "positive_frames": 0, "negative_frames": 0, "boxes": 0}
        for split in split_frames
    }
    preview_frames: list[tuple[int, str, object, list[list[float]]]] = []
    selected_for_preview = set()
    for split, frames in split_frames.items():
        ordered = sorted(frames)
        if ordered:
            for index in np.linspace(0, len(ordered) - 1, num=min(4, len(ordered)), dtype=int):
                selected_for_preview.add(ordered[int(index)])

    max_needed = max(max(frames) for frames in split_frames.values())
    frame_id = 0
    while frame_id <= max_needed:
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Video ended before frame {frame_id}")
        split = next((name for name, frames in split_frames.items() if frame_id in frames), None)
        if split is not None:
            stem = f"frame_{frame_id:06d}"
            image_path = args.output_dir / "images" / split / f"{stem}.jpg"
            label_path = args.output_dir / "labels" / split / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {image_path}")
            rows = [
                row
                for box in frame_boxes.get(frame_id, [])
                if (row := yolo_row(box, video_width, video_height)) is not None
            ]
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            stats[split]["frames"] += 1
            stats[split]["boxes"] += len(rows)
            stats[split]["positive_frames" if rows else "negative_frames"] += 1
            if frame_id in selected_for_preview:
                preview_frames.append((frame_id, split, image.copy(), frame_boxes.get(frame_id, [])))
        frame_id += 1
    capture.release()

    yaml_text = (
        f"path: {args.output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
    )
    (args.output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    tiles = []
    for frame_id, split, image, boxes in sorted(preview_frames):
        for box in boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(
            image,
            f"{split} frame {frame_id} boxes={len(boxes)}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
        scale = 720 / image.shape[1]
        tiles.append(cv2.resize(image, (720, int(image.shape[0] * scale))))
    if tiles:
        tile_height = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            canvas = np.full((tile_height, 720, 3), 255, dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        rows = []
        for index in range(0, len(padded), 2):
            chunk = padded[index:index + 2]
            if len(chunk) == 1:
                chunk.append(np.full_like(chunk[0], 255))
            rows.append(np.hstack(chunk))
        cv2.imwrite(str(args.output_dir / "label_contact_sheet.jpg"), np.vstack(rows))

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Round 1 human-reviewed seed for person-detector fine-tuning",
        "annotations": {
            "path": str(args.annotations_xml.resolve()),
            "sha256": sha256(args.annotations_xml),
        },
        "video": {
            "path": str(args.video.resolve()),
            "sha256": sha256(args.video),
            "width": video_width,
            "height": video_height,
            "annotated_frames": frame_count,
        },
        "split": {
            "train_half_open": [args.train_range.start, args.train_range.stop],
            "purge_half_open": [args.train_range.stop, args.val_range.start],
            "val_half_open": [args.val_range.start, args.val_range.stop],
        },
        "stats": stats,
        "class_names": ["person"],
        "limitations": [
            "Train and validation frames come from one recording; validation is a pipeline sanity check only.",
            "The purge interval reduces direct temporal adjacency but does not provide recording-level independence.",
            "Track IDs are not consumed by YOLO detection training.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
