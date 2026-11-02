"""Generate pedestrian track proposals for manual correction in CVAT.

The detector is COCO-pretrained (COCO class 0: person). Automatic tracks are
proposals, not ground truth. Outputs share the same integer track IDs:

* tracks.json: convenient project-native representation;
* gt/gt.txt: MOTChallenge format;
* annotations.xml: CVAT for video 1.1 format;
* preview.mp4: optional visual inspection video.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    local_model = Path(__file__).resolve().parents[1] / "models" / "tracking" / "yolo11l.pt"
    parser.add_argument("--model", default=str(local_model))
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path(__file__).with_name("botsort_reid.yaml"),
    )
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def video_metadata(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    if metadata["width"] <= 0 or metadata["height"] <= 0:
        raise RuntimeError(f"Invalid video dimensions: {metadata}")
    return metadata


def append_cvat_box(track_node: ET.Element, row: dict, outside: int) -> None:
    ET.SubElement(
        track_node,
        "box",
        {
            "frame": str(row["frame_id"]),
            "outside": str(outside),
            "occluded": "0",
            "keyframe": "1",
            "xtl": f'{row["bbox_xyxy"][0]:.2f}',
            "ytl": f'{row["bbox_xyxy"][1]:.2f}',
            "xbr": f'{row["bbox_xyxy"][2]:.2f}',
            "ybr": f'{row["bbox_xyxy"][3]:.2f}',
            "z_order": "0",
        },
    )


def write_cvat(path: Path, tracks: dict[int, list[dict]], video: Path, meta: dict) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta_node = ET.SubElement(root, "meta")
    task = ET.SubElement(meta_node, "task")
    ET.SubElement(task, "name").text = video.stem
    ET.SubElement(task, "size").text = str(meta["processed_frames"])
    ET.SubElement(task, "mode").text = "interpolation"
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "person"
    ET.SubElement(label, "color").text = "#ff00ff"
    ET.SubElement(meta_node, "source").text = str(video.resolve())

    for track_id, rows in sorted(tracks.items()):
        track_node = ET.SubElement(
            root, "track", {"id": str(track_id), "label": "person", "source": "auto"}
        )
        previous = None
        for row in rows:
            if previous is not None and row["frame_id"] > previous["frame_id"] + 1:
                outside_row = dict(previous)
                outside_row["frame_id"] += 1
                append_cvat_box(track_node, outside_row, outside=1)
            append_cvat_box(track_node, row, outside=0)
            previous = row
        if previous is not None and previous["frame_id"] + 1 < meta["processed_frames"]:
            outside_row = dict(previous)
            outside_row["frame_id"] += 1
            append_cvat_box(track_node, outside_row, outside=1)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.tracker.is_file():
        raise FileNotFoundError(args.tracker)

    try:
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime. Install with: pip install -r tracking/requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gt").mkdir(exist_ok=True)
    meta = video_metadata(args.video)
    if args.max_frames is not None:
        meta["frame_count"] = min(meta["frame_count"], args.max_frames)

    model = YOLO(args.model)
    result_stream = model.track(
        source=str(args.video),
        stream=True,
        persist=True,
        classes=[0],
        tracker=str(args.tracker),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    tracks: dict[int, list[dict]] = defaultdict(list)
    mot_lines: list[str] = []
    preview_writer = None
    preview_path = args.output_dir / "preview.mp4"
    if args.preview:
        preview_writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            meta["fps"],
            (meta["width"], meta["height"]),
        )

    processed_frames = 0
    for frame_id, result in enumerate(result_stream):
        if args.max_frames is not None and frame_id >= args.max_frames:
            break
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            for track_id, coords, confidence in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = [float(v) for v in coords]
                row = {
                    "frame_id": frame_id,
                    "track_id": int(track_id),
                    "label": "person",
                    "confidence": float(confidence),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "footpoint": [(x1 + x2) / 2.0, y2],
                }
                tracks[int(track_id)].append(row)
                # MOT uses one-based frames and x,y,w,h.
                mot_lines.append(
                    f"{frame_id + 1},{track_id},{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},{confidence:.6f},1,1"
                )
        if preview_writer is not None:
            preview_writer.write(result.plot())
        processed_frames = frame_id + 1

    if preview_writer is not None:
        preview_writer.release()
    meta["processed_frames"] = processed_frames

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Automatic proposals for human correction; not ground truth.",
        "video_path": str(args.video.resolve()),
        "video": meta,
        "method": {
            "detector": args.model,
            "detector_pretraining": "COCO, person class only",
            "tracker": "BoT-SORT with GMC and ReID",
            "tracker_config": str(args.tracker.resolve()),
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "device": args.device,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "tracks": [
            {"track_id": track_id, "label": "person", "boxes": rows}
            for track_id, rows in sorted(tracks.items())
        ],
    }
    (args.output_dir / "tracks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "gt" / "gt.txt").write_text(
        "\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8"
    )
    write_cvat(args.output_dir / "annotations.xml", tracks, args.video, meta)
    summary = {
        "processed_frames": processed_frames,
        "track_count": len(tracks),
        "box_count": sum(len(rows) for rows in tracks.values()),
        "tracks_at_least_60_frames": sum(len(rows) >= 60 for rows in tracks.values()),
        "outputs": ["tracks.json", "gt/gt.txt", "annotations.xml"]
        + (["preview.mp4"] if args.preview else []),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
