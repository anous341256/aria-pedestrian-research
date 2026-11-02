"""Create deterministic contact sheets for cross-model exclusive proposals."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def choose(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    chosen = []
    used_blocks = set()
    for row in rows:
        block = int(row["frame"]) // 30
        if block in used_blocks:
            continue
        chosen.append(row)
        used_blocks.add(block)
        if len(chosen) == count:
            return chosen
    for row in rows:
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == count:
                break
    return chosen


def crop_with_box(image: np.ndarray, row: dict, side: int = 512) -> np.ndarray:
    box = np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")])
    center_x, center_y = (box[:2] + box[2:]) / 2
    width, height = box[2:] - box[:2]
    extent = max(256, min(900, 2.7 * max(width, height)))
    left = int(round(center_x - extent / 2))
    top = int(round(center_y - extent / 2))
    right = left + int(round(extent))
    bottom = top + int(round(extent))
    patch = np.full((bottom - top, right - left, 3), 48, dtype=np.uint8)
    ix1, iy1 = max(left, 0), max(top, 0)
    ix2, iy2 = min(right, image.shape[1]), min(bottom, image.shape[0])
    if ix2 > ix1 and iy2 > iy1:
        patch[iy1 - top:iy2 - top, ix1 - left:ix2 - left] = image[iy1:iy2, ix1:ix2]
    patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_AREA)
    scale = side / (right - left)
    x1, y1, x2, y2 = (int(round(value)) for value in (
        (box[0] - left) * scale, (box[1] - top) * scale,
        (box[2] - left) * scale, (box[3] - top) * scale,
    ))
    cv2.rectangle(patch, (x1, y1), (x2, y2), (0, 0, 255), 3)
    title = f"f{row['frame']} {row['model']} c={float(row['confidence']):.2f} h={height:.0f} iou={float(row['nearest_other_iou']):.2f}"
    cv2.rectangle(patch, (0, 0), (side, 32), (0, 0, 0), -1)
    cv2.putText(patch, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA)
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--exclusive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into nonempty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.exclusive.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    models = sorted({row["model"] for row in all_rows})
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    try:
        for model in models:
            for size_name, sizes in (("tiny", {"h_lt_48"}),
                                     ("small", {"h_48_96"}),
                                     ("large", {"h_96_192", "h_ge_192"})):
                candidates = [row for row in all_rows if row["model"] == model
                              and row["height_bin"] in sizes
                              and float(row["nearest_other_iou"]) < 0.3]
                selected = choose(candidates, args.count,
                                  seed=19 if model == models[0] else 37)
                if not selected:
                    continue
                tiles = []
                for row in selected:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot read frame {row['frame']}")
                    tiles.append(crop_with_box(image, row))
                while len(tiles) % 3:
                    tiles.append(np.full((512, 512, 3), 48, dtype=np.uint8))
                montage = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
                path = args.output_dir / f"{model}_{size_name}.jpg"
                cv2.imwrite(str(path), montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with (args.output_dir / f"{model}_{size_name}_cases.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
                    writer.writeheader()
                    writer.writerows(selected)
                print(f"{path}: {len(selected)} cases")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
