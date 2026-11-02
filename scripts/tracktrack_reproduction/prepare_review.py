"""Export online MOT proposals to CVAT and structural checks, not accuracy scores.

Reuses this project's CVAT writer; explicit outside markers preserve empty gaps.
Does not import into CVAT or change any source annotations.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tracking"))
from auto_annotate_people import write_cvat, video_metadata
from audit_cvat_video_annotations import parse_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = video_metadata(args.video)
    meta["processed_frames"] = meta["frame_count"]
    by_id, by_frame = collections.defaultdict(list), collections.defaultdict(list)
    seen, clipped = set(), 0
    with args.mot.open(newline="", encoding="utf-8") as handle:
        for line in csv.reader(handle):
            if not line:
                continue
            f, tid = int(line[0]) - 1, int(line[1])
            x, y, w, h, score = map(float, line[2:7])
            if not (0 <= f < meta["frame_count"] and tid > 0 and w > 0 and h > 0):
                raise ValueError(f"Invalid MOT row: {line}")
            if (f, tid) in seen:
                raise ValueError(f"Duplicate frame/ID row: {(f, tid)}")
            if not np.isfinite([x, y, w, h, score]).all():
                raise ValueError(f"Nonfinite MOT row: {line}")
            seen.add((f, tid))
            raw = np.asarray([x, y, x + w, y + h])
            box = np.clip(raw, [0, 0, 0, 0],
                          [meta["width"], meta["height"], meta["width"], meta["height"]])
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Track fully outside image: {line}")
            clipped += int(not np.array_equal(raw, box))
            row = {"frame_id": f, "bbox_xyxy": box.tolist(), "confidence": score,
                   "track_id": tid}
            by_id[tid].append(row)
            by_frame[f].append(row)
    for rows in by_id.values():
        rows.sort(key=lambda row: row["frame_id"])
    xml_path = args.output / "annotations_cvat.xml"
    write_cvat(xml_path, by_id, args.video, meta)
    tree = ET.parse(xml_path)
    original_size = ET.SubElement(tree.getroot().find("meta"), "original_size")
    ET.SubElement(original_size, "width").text = str(meta["width"])
    ET.SubElement(original_size, "height").text = str(meta["height"])
    ET.indent(tree.getroot(), space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    # Verify the CVAT interpolation semantics do not invent visible gap boxes.
    parsed = parse_xml(xml_path)
    actual = {(f, tid) for tid, track in parsed["tracks"].items()
              for f in track["visible"]}
    if actual != seen:
        raise AssertionError(f"CVAT roundtrip differs: {len(actual ^ seen)} rows")
    lengths = [len(rows) for rows in by_id.values()]
    gaps = sum(sum(b["frame_id"] > a["frame_id"] + 1 for a, b in zip(rows, rows[1:]))
               for rows in by_id.values())
    # These are counts, NOT ID switches, fragments against GT, recall, or precision.
    stats = {"video": str(args.video.resolve()), "mot": str(args.mot.resolve()),
             "metadata": meta, "output_boxes": len(seen), "output_track_ids": len(by_id),
             "frames_with_tracks": len(by_frame),
             "median_visible_frames_per_id": float(np.median(lengths)) if lengths else 0,
             "ids_visible_at_most_10_frames": sum(n <= 10 for n in lengths),
             "same_id_reappearance_gaps": gaps,
             "cvat_boxes_clipped_to_image": clipped,
             "cvat_roundtrip_visible_frame_id_pairs_equal": True,
             "warning": "Unreviewed automatic person proposals. Counts are not accuracy metrics. "
                        "No interpolation through missing detections. Raw MOT coordinates preserved."}
    (args.output / "review_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with (args.output / "track_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "first_frame_0based", "last_frame_0based", "visible_frames", "gaps"])
        for tid, rows in sorted(by_id.items()):
            writer.writerow([tid, rows[0]["frame_id"], rows[-1]["frame_id"], len(rows),
                             sum(b["frame_id"] > a["frame_id"] + 1 for a, b in zip(rows, rows[1:]))])
    frame_ids = sorted(set([min(meta["frame_count"] - 1, int(t * meta["fps"]))
                            for t in (5, 20, 35, 50)]))
    cap = cv2.VideoCapture(str(args.video))
    tiles = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Preview decode failed: {fid}")
        for row in by_frame[fid]:
            x1, y1, x2, y2 = map(round, row["bbox_xyxy"])
            tid = row["track_id"]
            color = (50 + (tid * 73) % 206, 50 + (tid * 151) % 206,
                     50 + (tid * 199) % 206)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, str(tid), (x1, max(25, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Frame {fid+1} | {fid/meta['fps']:.1f}s | {len(by_frame[fid])} tracks",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
        cv2.imwrite(str(args.output / f"review_frame_{fid+1:06}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        tiles.append(cv2.resize(frame, (960, round(960 * meta["height"] / meta["width"]))))
    cap.release()
    if len(tiles) == 4:
        cv2.imwrite(str(args.output / "review_contact_sheet.jpg"),
                    np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
