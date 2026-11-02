"""Compare two CVAT video-track proposals without treating either as truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from audit_cvat_video_annotations import iou, parse_xml


def greedy_match(left, right, threshold):
    candidates = sorted(
        ((iou(a, b), i, j) for i, a in enumerate(left) for j, b in enumerate(right)),
        reverse=True,
    )
    used_left, used_right, matches = set(), set(), []
    for overlap, i, j in candidates:
        if overlap < threshold:
            break
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append((i, j, overlap))
    return matches, len(left) - len(used_left), len(right) - len(used_right)


def draw(image, boxes, color):
    import cv2

    for box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)


def make_sheet(video, rows, left_frames, right_frames, output_path, title_left, title_right):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    tiles = []
    for row in rows:
        frame_id = row["frame"]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        if not ok:
            continue
        left = image.copy()
        right = image.copy()
        draw(left, left_frames.get(frame_id, []), (0, 0, 255))
        draw(right, right_frames.get(frame_id, []), (0, 255, 0))
        cv2.putText(left, f"{title_left} f={frame_id} n={row['left_boxes']}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(right, f"{title_right} f={frame_id} n={row['right_boxes']}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        pair = np.hstack([left, right])
        scale = 1280 / pair.shape[1]
        tiles.append(cv2.resize(pair, (1280, int(pair.shape[0] * scale))))
    capture.release()
    if tiles:
        cv2.imwrite(str(output_path), np.vstack(tiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_xml", type=Path)
    parser.add_argument("right_xml", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    args = parser.parse_args()

    left = parse_xml(args.left_xml)
    right = parse_xml(args.right_xml)
    if left["size"] != right["size"]:
        raise ValueError("annotation frame counts differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def frame_boxes(dataset):
        result = {frame: [] for frame in range(dataset["size"])}
        for track in dataset["tracks"].values():
            for frame, box in track["visible"].items():
                result[frame].append(box)
        return result

    left_frames, right_frames = frame_boxes(left), frame_boxes(right)
    rows = []
    for frame in range(left["size"]):
        matches, left_only, right_only = greedy_match(left_frames[frame], right_frames[frame], args.iou)
        rows.append({
            "frame": frame,
            "left_boxes": len(left_frames[frame]),
            "right_boxes": len(right_frames[frame]),
            "matched": len(matches),
            "left_only": left_only,
            "right_only": right_only,
            "disagreement": left_only + right_only,
        })
    with (args.output_dir / "frame_disagreements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: (row["disagreement"], row["left_only"], row["right_only"]), reverse=True)
    uniform_ids = [round(index * (left["size"] - 1) / 11) for index in range(12)]
    uniform_rows = [rows[frame] for frame in uniform_ids]
    make_sheet(args.video, ranked[:12], left_frames, right_frames, args.output_dir / "top_disagreements.jpg", args.left_name, args.right_name)
    make_sheet(args.video, uniform_rows, left_frames, right_frames, args.output_dir / "uniform_comparison.jpg", args.left_name, args.right_name)

    summary = {
        "frame_count": left["size"],
        "iou_match_threshold": args.iou,
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_boxes": sum(row["left_boxes"] for row in rows),
        "right_boxes": sum(row["right_boxes"] for row in rows),
        "matched_boxes": sum(row["matched"] for row in rows),
        "left_only_boxes": sum(row["left_only"] for row in rows),
        "right_only_boxes": sum(row["right_only"] for row in rows),
        "frames_with_disagreement": sum(row["disagreement"] > 0 for row in rows),
        "warning": "Neither proposal is ground truth; unmatched boxes require human review.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
