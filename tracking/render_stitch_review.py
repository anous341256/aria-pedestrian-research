"""Render before/after crops for every accepted offline tracklet link."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_cvat_video_annotations import parse_xml


def read_frame(capture: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id}")
    return frame


def context_crop(image: np.ndarray, box: list[float], size=(560, 300)) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box_h = max(y2 - y1, 30)
    crop_h = min(height, max(240, int(box_h * 3.2)))
    crop_w = min(width, int(crop_h * size[0] / size[1]))
    left = max(0, min(width - crop_w, int(cx - crop_w / 2)))
    top = max(0, min(height - crop_h, int(cy - crop_h / 2)))
    crop = image[top : top + crop_h, left : left + crop_w].copy()
    shifted = [x1 - left, y1 - top, x2 - left, y2 - top]
    cv2.rectangle(
        crop,
        (int(shifted[0]), int(shifted[1])),
        (int(shifted[2]), int(shifted[3])),
        (0, 255, 255),
        max(2, crop_h // 160),
    )
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = parse_xml(args.annotations)
    links = json.loads(args.manifest.read_text(encoding="utf-8"))["links"]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    rows = []
    for index, link in enumerate(links, start=1):
        frame_a, frame_b = link["from_frame"], link["to_frame"]
        box_a = dataset["tracks"][link["from_track"]]["visible"][frame_a]
        box_b = dataset["tracks"][link["to_track"]]["visible"][frame_b]
        left = context_crop(read_frame(capture, frame_a), box_a)
        right = context_crop(read_frame(capture, frame_b), box_b)
        header = np.full((70, 1120, 3), 245, dtype=np.uint8)
        title = (
            f"#{index:02d}  ID {link['from_track']} -> {link['to_track']}  "
            f"frames {frame_a}->{frame_b} gap={link['missing_frames']}  "
            f"sim={link['appearance_similarity']:.3f} gmc={link['gmc_normalized_distance']:.3f}"
        )
        cv2.putText(header, title, (16, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
        row = np.vstack([header, np.hstack([left, right])])
        cv2.imwrite(str(args.output_dir / f"link_{index:02d}_{link['from_track']}_{link['to_track']}.jpg"), row)
        rows.append(row)
    capture.release()
    if rows:
        cv2.imwrite(str(args.output_dir / "contact_sheet.jpg"), np.vstack(rows))
    print(json.dumps({"links": len(links), "output": str(args.output_dir / "contact_sheet.jpg")}, indent=2))


if __name__ == "__main__":
    main()
