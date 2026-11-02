#!/usr/bin/env python3
"""Export cached detector shapes versus real TrackTrack tracks for CVAT 1.1.

Reads only existing experiment artifacts. No inference, server calls, or uploads.
Image-level detector boxes have no cross-frame identity. Explicit outside markers
preserve all missing intervals in TrackTrack trajectories. Clipping is export-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tracking"))
from audit_cvat_video_annotations import expand_track


FORMAT_DOC = "https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/"
PARSER_URL = "https://github.com/cvat-ai/cvat/blob/develop/cvat/apps/dataset_manager/formats/cvat.py#L1302-L1549"
MATCHER_URL = "https://github.com/cvat-ai/cvat/blob/develop/cvat/apps/dataset_manager/bindings.py#L2352-L2369"


def text_attribute(name, mutable=True):
    # The current CVAT Raw label editor requires a non-empty values array even
    # for unrestricted text attributes; one empty string is its accepted default.
    return {"name": name, "input_type": "text", "mutable": mutable,
            "default_value": "", "values": [""]}


def label_spec(name, color, with_track_id=False):
    attrs = [text_attribute("score"), text_attribute("source_frame_1based")]
    if with_track_id:
        attrs.append(text_attribute("track_id", False))
    return {"name": name, "color": color, "type": "rectangle", "attributes": attrs}


LABELS = [label_spec("detector_person", "#f4c542"), label_spec("tracktrack_person", "#00bcd4", True)]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def clip_box(raw, width, height):
    box = np.asarray(raw, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"Invalid box: {raw}")
    clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Box has no positive area after export clipping: {raw}")
    return clipped.tolist(), bool(not np.array_equal(box, clipped))


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS))}
        if meta != {"width": 2560, "height": 1920, "frame_count": 600, "fps": 10.0}:
            raise ValueError(f"This comparison targets the 600-frame 10-FPS 2560x1920 clip, got {meta}")
        for frame in (0, 300, 599):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image.shape[:2] != (1920, 2560):
                raise ValueError(f"Video verification failed at zero-based frame {frame}")
    finally:
        cap.release()
    return meta


def read_inputs(run_dir, meta):
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "complete" or summary["result"]["frames"] != meta["frame_count"]:
        raise ValueError("Source run is not a complete 600-frame run")
    recorded = summary["video_metadata"]
    for key in ("width", "height", "fps"):
        if recorded[key] != meta[key]:
            raise ValueError(f"Source/video {key} mismatch")
    detector = {"80": defaultdict(list), "95": defaultdict(list)}
    clips = {"80": 0, "95": 0, "tracking": 0}
    names = {p.name for p in (run_dir / "detections").glob("*.npz")}
    if names != {f"{i:06d}.npz" for i in range(1, 601)}:
        raise ValueError("Detection cache must contain exactly frames 000001 through 000600")
    for frame in range(600):
        with np.load(run_dir / "detections" / f"{frame + 1:06d}.npz", allow_pickle=False) as cache:
            if int(cache["frame_id"]) != frame + 1:
                raise ValueError("Cache frame ID mismatch")
            for branch in ("80", "95"):
                dets = cache[f"det{branch}"]
                if dets.ndim != 2 or dets.shape[1] != 6 or not np.isfinite(dets).all():
                    raise ValueError(f"Invalid detector data at frame {frame + 1}")
                for row_index, det in enumerate(dets):
                    if det[5] != 0:
                        raise ValueError("Expected the single-class author pedestrian detector")
                    box, clipped = clip_box(det[:4], meta["width"], meta["height"])
                    detector[branch][frame].append({"frame": frame, "score": float(det[4]),
                                                    "box": box, "raw_box": det[:4].astype(float).tolist(),
                                                    "row_index": row_index})
                    clips[branch] += int(clipped)
    tracks = defaultdict(list)
    seen = set()
    with (run_dir / "tracks.txt").open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            if not fields:
                continue
            frame, track_id = int(fields[0]) - 1, int(fields[1])
            x, y, w, h, score = map(float, fields[2:7])
            if not 0 <= frame < 600 or track_id <= 0 or w <= 0 or h <= 0 or (frame, track_id) in seen:
                raise ValueError(f"Invalid/duplicate MOT row: {fields}")
            if not np.isfinite([x, y, w, h, score]).all():
                raise ValueError("Nonfinite MOT row")
            box, clipped = clip_box([x, y, x + w, y + h], meta["width"], meta["height"])
            tracks[track_id].append({"frame": frame, "score": score, "box": box})
            seen.add((frame, track_id))
            clips["tracking"] += int(clipped)
    for rows in tracks.values():
        rows.sort(key=lambda row: row["frame"])
    counts = {"detector80": sum(map(len, detector["80"].values())),
              "detector95": sum(map(len, detector["95"].values())), "tracking": len(seen)}
    if counts["detector80"] != 5513 or counts["tracking"] != 2066:
        raise ValueError(f"Expected 5513 primary detections vs 2066 tracked boxes, got {counts}")
    if counts["detector80"] != summary["result"]["raw_det80"] or counts["detector95"] != summary["result"]["raw_det95"]:
        raise ValueError("Detector totals differ from source summary")
    return detector, tracks, counts, clips, summary["detector"]


def annotation_root(meta, video, labels):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    metadata = ET.SubElement(root, "meta")
    task = ET.SubElement(metadata, "task")
    for name, value in {"name": video.stem, "size": meta["frame_count"], "mode": "interpolation",
                        "start_frame": 0, "stop_frame": meta["frame_count"] - 1,
                        "frame_filter": "", "flipped": "False", "source": video.name}.items():
        ET.SubElement(task, name).text = str(value)
    original_size = ET.SubElement(task, "original_size")
    for name in ("width", "height"):
        ET.SubElement(original_size, name).text = str(meta[name])
    label_nodes = ET.SubElement(task, "labels")
    for spec in labels:
        label = ET.SubElement(label_nodes, "label")
        for name in ("name", "color", "type"):
            ET.SubElement(label, name).text = spec[name]
        attributes = ET.SubElement(label, "attributes")
        for attr in spec["attributes"]:
            node = ET.SubElement(attributes, "attribute")
            for key in ("name", "mutable", "input_type", "default_value", "values"):
                value = attr[key]
                if key == "mutable":
                    value = str(value).lower()
                elif key == "values":
                    value = "\n".join(value)
                ET.SubElement(node, key).text = value
    return root


def add_box(parent, row, label=None, outside=None, track_id=None):
    coords = {key: f"{value:.6f}" for key, value in zip(("xtl", "ytl", "xbr", "ybr"), row["box"])}
    attrs = {**coords, "occluded": "0", "z_order": "0"}
    if label is not None:
        attrs.update(label=label, source="auto")
    if outside is not None:
        attrs.update(frame=str(row["frame"]), outside=str(int(outside)), keyframe="1")
    node = ET.SubElement(parent, "box", attrs)
    ET.SubElement(node, "attribute", name="score").text = format(row["score"], ".9g")
    ET.SubElement(node, "attribute", name="source_frame_1based").text = str(row["frame"] + 1)
    if track_id is not None:
        ET.SubElement(node, "attribute", name="track_id").text = str(track_id)


def add_detector_shapes(root, detections, meta, label):
    # Images/boxes import as standalone shapes, not pseudo-tracks. The official
    # importer falls back to this image id for video frame matching.
    for frame in range(meta["frame_count"]):
        image = ET.SubElement(root, "image", id=str(frame), name=f"frame_{frame:06d}.png",
                              width=str(meta["width"]), height=str(meta["height"]))
        for row in detections.get(frame, []):
            add_box(image, row, label=label)


def add_real_tracks(root, tracks, meta, label):
    for track_id, rows in sorted(tracks.items()):
        node = ET.SubElement(root, "track", id=str(track_id), label=label, source="auto")
        previous = None
        for row in rows:
            if previous is not None and row["frame"] > previous["frame"] + 1:
                add_box(node, {**previous, "frame": previous["frame"] + 1}, outside=True, track_id=track_id)
            add_box(node, row, outside=False, track_id=track_id)
            previous = row
        if previous is not None and previous["frame"] + 1 < meta["frame_count"]:
            add_box(node, {**previous, "frame": previous["frame"] + 1}, outside=True, track_id=track_id)


def save_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def verify_xml(path, detections, tracks, detector_label, track_label):
    root = ET.parse(path).getroot()
    image_nodes = root.findall("image")
    shape_count = 0
    if detections is not None:
        if [int(node.attrib["id"]) for node in image_nodes] != list(range(600)):
            raise AssertionError("Shape frame coverage is not exactly 0..599")
        for image in image_nodes:
            frame = int(image.attrib["id"])
            nodes, expected = image.findall("box"), detections.get(frame, [])
            if len(nodes) != len(expected):
                raise AssertionError("Per-frame shape count changed")
            for node, row in zip(nodes, expected):
                if node.attrib["label"] != detector_label or "frame" in node.attrib or "track_id" in node.attrib:
                    raise AssertionError("Detector shape was assigned track semantics")
                attributes = {child.attrib["name"]: child.text for child in node.findall("attribute")}
                if "track_id" in attributes or int(attributes["source_frame_1based"]) != frame + 1:
                    raise AssertionError("Detector ID/frame attributes are wrong")
                box = [float(node.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
                np.testing.assert_allclose(box, row["box"], rtol=0, atol=5.01e-7)
                np.testing.assert_allclose(float(attributes["score"]), row["score"], rtol=1e-8, atol=1e-8)
            shape_count += len(nodes)
    elif image_nodes:
        raise AssertionError("Unexpected detector shapes in track-only XML")
    visible_pairs, outside_count = set(), 0
    for track in root.findall("track"):
        tid = int(track.attrib["id"])
        if tracks is None or track.attrib["label"] != track_label or tid not in tracks:
            raise AssertionError("Unexpected track")
        keys = []
        for box in track.findall("box"):
            frame = int(box.attrib["frame"])
            outside = box.attrib["outside"] == "1"
            attrs = {child.attrib["name"]: child.text for child in box.findall("attribute")}
            if int(attrs["track_id"]) != tid or int(attrs["source_frame_1based"]) != frame + 1:
                raise AssertionError("Track attributes disagree with identity/frame")
            if not 0 <= frame < 600:
                raise AssertionError("Track keyframe outside clip range")
            keys.append({"frame": frame, "outside": outside,
                         "box": [float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]})
            outside_count += int(outside)
        visible = expand_track(keys, 599)
        expected = {row["frame"]: row for row in tracks[tid]}
        if set(visible) != set(expected):
            raise AssertionError(f"Track {tid} interpolates across a missing interval")
        for frame, box in visible.items():
            np.testing.assert_allclose(box, expected[frame]["box"], rtol=0, atol=5.01e-7)
            visible_pairs.add((frame, tid))
    expected_pairs = {(row["frame"], tid) for tid, rows in (tracks or {}).items() for row in rows}
    if visible_pairs != expected_pairs:
        raise AssertionError("Visible tracking frame/ID pairs changed")
    return {"standalone_detector_shapes": shape_count, "real_track_ids": len(root.findall("track")),
            "visible_track_boxes": len(visible_pairs), "outside_markers": outside_count,
            "no_detector_cross_frame_ids": True, "track_visible_frame_id_pairs_exact": True}


def export(args):
    args.run_dir, args.video, args.output = args.run_dir.resolve(), args.video.resolve(), args.output.resolve()
    # A labels.json may already have been prepared while the CVAT task was being
    # created. Preserve it unless it matches the exact requested label schema.
    if args.output.exists():
        extras = [p.name for p in args.output.iterdir() if p.name != "labels.json"]
        if extras:
            raise FileExistsError(f"Refusing to overwrite existing comparison artifacts: {extras}")
        label_path = args.output / "labels.json"
        if label_path.exists() and json.loads(label_path.read_text(encoding="utf-8")) != LABELS:
            raise ValueError("Existing labels.json differs; do not overwrite user label edits")
    meta = video_metadata(args.video)
    original_mot_hash = digest(args.run_dir / "tracks.txt")
    detector, tracks, counts, clips, detector_parameters = read_inputs(args.run_dir, meta)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "labels.json", LABELS)
    generic = [label_spec("person", "#00bcd4", True)]
    write_json(args.output / "labels_person.json", generic)
    outputs = {}
    definitions = [
        ("comparison_detector_vs_tracktrack.xml", LABELS, detector["80"], tracks, "detector_person", "tracktrack_person"),
        ("detector_only_person.xml", [label_spec("person", "#f4c542")], detector["80"], None, "person", None),
        ("tracktrack_only_person.xml", generic, None, tracks, None, "person"),
        ("local_only_nms095_person.xml", [label_spec("person", "#f4c542")], detector["95"], None, "person", None),
    ]
    for filename, labels, det, trk, det_label, trk_label in definitions:
        root = annotation_root(meta, args.video, labels)
        if det is not None:
            add_detector_shapes(root, det, meta, det_label)
        if trk is not None:
            add_real_tracks(root, trk, meta, trk_label)
        path = args.output / filename
        save_xml(path, root)
        outputs[filename] = {"sha256": digest(path), **verify_xml(path, det, trk, det_label, trk_label)}
    with (args.output / "frame_comparison_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cvat_frame_0based", "source_frame_1based", "video_seconds", "detector_nms080_shapes", "tracktrack_boxes", "local_only_nms095_candidates"])
        track_counts = defaultdict(int)
        for rows in tracks.values():
            for row in rows:
                track_counts[row["frame"]] += 1
        for frame in range(600):
            writer.writerow([frame, frame + 1, f"{frame / 10:.1f}", len(detector["80"].get(frame, [])),
                             track_counts[frame], len(detector["95"].get(frame, []))])
    if digest(args.run_dir / "tracks.txt") != original_mot_hash:
        raise AssertionError("Original MOT file changed during export")
    review_examples = [{"cvat_frame_0based": frame, "source_frame_1based": frame + 1,
                        "video_seconds": frame / 10, "detector80_boxes": len(detector["80"].get(frame, [])),
                        "detector80_score_above_0_7": sum(row["score"] > .7 for row in detector["80"].get(frame, [])),
                        "tracktrack_boxes": track_counts[frame]} for frame in (50, 75)]
    result = {"status": "complete", "video": {"path": str(args.video), "sha256": digest(args.video), **meta},
              "source_run": str(args.run_dir), "source_mot_sha256": original_mot_hash,
              "detector_parameters": detector_parameters, "counts": counts, "original_track_ids": len(tracks),
              "export_only_clipped_boxes": clips, "original_mot_preserved": True,
              "frame_convention": "CVAT image IDs and track frame fields are 0..599; source_frame_1based is cache/MOT frame 1..600; t=CVAT_frame/10",
              "detection_semantics": "det80, all cached boxes; no extra score/area/aspect/crop filtering; standalone image shapes without cross-frame IDs",
              "track_semantics": "Original online TrackTrack MOT output, real identities with explicit outside markers; no interpolation through missing output frames",
              "comparison_caution": "TrackTrack also uses high-confidence candidates from NMS 0.95; cyan boxes need not have a matching NMS 0.80 yellow box. NMS 0.95 XML is local-only, not part of the main comparison.",
              "not_accuracy_metrics": True, "review_examples": review_examples,
              "format_evidence": [FORMAT_DOC, PARSER_URL, MATCHER_URL], "files": outputs}
    write_json(args.output / "export_summary.json", result)
    readme = (
        "# Detector / TrackTrack CVAT comparison\n\n"
        "Import `comparison_detector_vs_tracktrack.xml` using CVAT 1.1 into a NEW task containing the unchanged input video. "
        "Use `labels.json` in the Raw label editor. Do not overwrite reviewed annotations.\n\n"
        "- Yellow `detector_person`: 5,513 independent per-frame shapes, score threshold 0.1 and NMS 0.80. "
        "No pedestrian identities are assigned to these detections.\n"
        "- Cyan `tracktrack_person`: 2,066 visible boxes in 259 original tracker identities; `track_id` preserves the original MOT ID. "
        "Explicit outside markers keep missing intervals invisible.\n"
        "- CVAT frame 0 corresponds to cache/MOT source frame 1. All 600 frames use the same 2560x1920, 10-FPS video.\n"
        "- Boxes are clipped only in these CVAT exports. Original caches and MOT remain unchanged.\n"
        "- TrackTrack also consumes high-confidence candidates from NMS 0.95. A cyan box without a yellow match is not automatically an error. "
        "`local_only_nms095_person.xml` is separate and must NOT be added to the main comparison automatically.\n\n"
        "Independent alternatives: `detector_only_person.xml` and `tracktrack_only_person.xml`, both with label `person`. "
        "`labels_person.json` is a compatible label definition for either alternative.\n\n"
        "Suggested inspection points: CVAT frame 50 (5.0 s, cache frame 51) has 6 primary detections, "
        "4 with score >0.7, but no tracker output; frame 75 (7.5 s, cache frame 76) has 5 detections, "
        "3 with score >0.7, but no tracker output. These are review prompts, not evidence that association "
        "alone caused the missing output. Inspect confirmation, gating, CMC and output filtering as well.\n\n"
        "Detection boxes are serialized as `<image><box>` shapes, trajectories as `<track><box>` with outside markers. "
        "The official CVAT importer handles both in one document and resolves image IDs as video frame indices. "
        "Do not convert detector row numbers to person IDs.\n\n"
        f"Format: {FORMAT_DOC}\nParser: {PARSER_URL}\nFrame matching: {MATCHER_URL}\n\n"
        "Counts are inspection aids, not precision/recall, identity-switch counts or accuracy measurements.\n"
    )
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    export(p.parse_args())
