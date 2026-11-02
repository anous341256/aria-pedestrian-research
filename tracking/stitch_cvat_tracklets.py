"""Offline one-to-one stitching of fragmented CVAT pedestrian tracks.

The linker only joins non-overlapping tracks. Candidate links are gated by the
temporal gap, scale change, normalized image displacement and endpoint visual
similarity. Appearance descriptors come from a local Ultralytics classification
model and are averaged over several crops at each track endpoint.

This is a proposal generator. Use a corrected development clip to choose the
thresholds, then keep them fixed on unseen recordings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_cvat_video_annotations import parse_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reid-model", type=Path, required=True)
    parser.add_argument(
        "--reid-backend", choices=("ultralytics", "torchreid", "boxmot"), default="ultralytics"
    )
    parser.add_argument("--reid-model-name", default="osnet_x0_25")
    parser.add_argument("--device", default="0")
    parser.add_argument("--endpoint-samples", type=int, default=4)
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.70)
    parser.add_argument("--max-size-ratio", type=float, default=2.5)
    parser.add_argument("--base-spatial-gate", type=float, default=1.75)
    parser.add_argument("--spatial-gate-per-frame", type=float, default=0.20)
    parser.add_argument("--appearance-weight", type=float, default=0.60)
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--gap-weight", type=float, default=0.10)
    parser.add_argument("--size-weight", type=float, default=0.05)
    parser.add_argument("--use-gmc", action="store_true")
    parser.add_argument("--gmc-weight", type=float, default=0.20)
    parser.add_argument("--gmc-base-gate", type=float, default=0.80)
    parser.add_argument("--gmc-gate-per-frame", type=float, default=0.12)
    parser.add_argument("--max-cost", type=float, default=0.55)
    return parser.parse_args()


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def height(box: list[float]) -> float:
    return max(1.0, box[3] - box[1])


def crop_box(image: np.ndarray, box: list[float]) -> np.ndarray | None:
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = 0.04 * max(1.0, x2 - x1)
    pad_y = 0.03 * max(1.0, y2 - y1)
    x1 = max(0, int(math.floor(x1 - pad_x)))
    y1 = max(0, int(math.floor(y1 - pad_y)))
    x2 = min(image_w, int(math.ceil(x2 + pad_x)))
    y2 = min(image_h, int(math.ceil(y2 + pad_y)))
    if x2 - x1 < 6 or y2 - y1 < 12:
        return None
    return image[y1:y2, x1:x2].copy()


def endpoint_frames(frames: list[int], count: int) -> tuple[list[int], list[int]]:
    if len(frames) <= count:
        return frames, frames
    return frames[:count], frames[-count:]


def extract_endpoint_embeddings(
    dataset: dict,
    video_path: Path,
    model_path: Path,
    device: str,
    samples: int,
    backend: str,
    model_name: str,
) -> dict[int, dict[str, np.ndarray]]:
    requests: dict[int, list[tuple[int, str, list[float]]]] = defaultdict(list)
    for track_id, track in dataset["tracks"].items():
        frames = sorted(track["visible"])
        if not frames:
            continue
        starts, ends = endpoint_frames(frames, samples)
        for frame in starts:
            requests[frame].append((track_id, "start", track["visible"][frame]))
        for frame in ends:
            requests[frame].append((track_id, "end", track["visible"][frame]))

    crops: list[np.ndarray] = []
    keys: list[tuple[int, str]] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_id = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        for track_id, endpoint, box in requests.get(frame_id, []):
            crop = crop_box(image, box)
            if crop is not None:
                crops.append(crop)
                keys.append((track_id, endpoint))
        frame_id += 1
    capture.release()

    grouped: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    batch_size = 128
    if backend == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        for start in range(0, len(crops), batch_size):
            outputs = model.embed(
                crops[start : start + batch_size],
                imgsz=224,
                device=device,
                verbose=False,
            )
            for key, output in zip(keys[start : start + batch_size], outputs):
                vector = output.detach().float().cpu().numpy()
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                grouped[key].append(vector)
    elif backend == "torchreid":
        # The official project contains an optional Cython rank evaluator whose
        # Windows build requires MSVC.  Feature extraction does not need it, so
        # use the vendored source tree directly and keep this pipeline portable.
        source_root = Path(__file__).resolve().parents[1] / "tools" / "deep-person-reid"
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        import torchreid

        reid_device = "cuda" if str(device).lower() != "cpu" else "cpu"
        extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=str(model_path),
            device=reid_device,
        )
        for start in range(0, len(crops), batch_size):
            rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops[start : start + batch_size]]
            outputs = extractor(rgb_crops).detach().float().cpu().numpy()
            for key, vector in zip(keys[start : start + batch_size], outputs):
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                grouped[key].append(vector)
    else:
        import torch
        from boxmot.reid import create_reid_encoder
        from boxmot.structures import Boxes, Detections, Frame

        encoder = create_reid_encoder(
            model_path,
            device="cuda:0" if str(device).lower() != "cpu" else "cpu",
            precision="fp16" if str(device).lower() != "cpu" else "fp32",
        )
        for start in range(0, len(crops), batch_size):
            frames = []
            detections = []
            for offset, crop in enumerate(crops[start : start + batch_size]):
                height_px, width_px = crop.shape[:2]
                sample_id = f"endpoint:{start + offset}"
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                frames.append(
                    Frame(
                        image=torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
                        sample_id=sample_id,
                        sequence_id="endpoint-crops",
                        frame_index=start + offset,
                    )
                )
                detections.append(
                    Detections(
                        geometry=Boxes(torch.tensor([[0, 0, width_px, height_px]], dtype=torch.float32)),
                        scores=torch.tensor([1.0], dtype=torch.float32),
                        class_ids=torch.tensor([0], dtype=torch.int64),
                        sample_id=sample_id,
                    )
                )
            outputs = encoder.encode(frames, detections)
            for key, output in zip(keys[start : start + batch_size], outputs):
                vector = output[0].detach().float().cpu().numpy()
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                grouped[key].append(vector)

    result: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for (track_id, endpoint), vectors in grouped.items():
        vector = np.mean(vectors, axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        result[track_id][endpoint] = vector
    return dict(result)


def estimate_pairwise_gmc(video_path: Path, scale: float = 0.25) -> list[np.ndarray]:
    """Estimate frame-to-frame background motion as full-resolution affine matrices."""
    capture = cv2.VideoCapture(str(video_path))
    ok, previous = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    previous = cv2.resize(previous, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    transforms: list[np.ndarray] = []
    while True:
        ok, current = capture.read()
        if not ok:
            break
        current = cv2.resize(current, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=1200, qualityLevel=0.01, minDistance=8, blockSize=5
        )
        affine = None
        if points is not None and len(points) >= 12:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if tracked is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                if int(valid.sum()) >= 12:
                    affine, _ = cv2.estimateAffinePartial2D(
                        points.reshape(-1, 2)[valid],
                        tracked.reshape(-1, 2)[valid],
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.5,
                        maxIters=2000,
                        confidence=0.995,
                    )
        matrix = np.eye(3, dtype=np.float64)
        if affine is not None and np.isfinite(affine).all():
            matrix[:2, :2] = affine[:, :2]
            matrix[0, 2] = affine[0, 2] / scale
            matrix[1, 2] = affine[1, 2] / scale
        transforms.append(matrix)
        previous_gray = current_gray
    capture.release()
    return transforms


def compose_gmc(transforms: list[np.ndarray], frame_from: int, frame_to: int) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    for frame in range(frame_from, frame_to):
        if frame >= len(transforms):
            break
        matrix = transforms[frame] @ matrix
    return matrix


def candidate_cost(
    left_id: int,
    right_id: int,
    dataset: dict,
    embeddings: dict[int, dict[str, np.ndarray]],
    gmc_transforms: list[np.ndarray] | None,
    args: argparse.Namespace,
) -> tuple[float, dict] | None:
    left_frames = sorted(dataset["tracks"][left_id]["visible"])
    right_frames = sorted(dataset["tracks"][right_id]["visible"])
    if not left_frames or not right_frames or left_frames[-1] >= right_frames[0]:
        return None
    gap = right_frames[0] - left_frames[-1] - 1
    if gap > args.max_gap:
        return None

    left_box = dataset["tracks"][left_id]["visible"][left_frames[-1]]
    right_box = dataset["tracks"][right_id]["visible"][right_frames[0]]
    size_ratio = max(height(left_box), height(right_box)) / min(height(left_box), height(right_box))
    if size_ratio > args.max_size_ratio:
        return None
    normalized_distance = math.dist(center(left_box), center(right_box)) / (
        0.5 * (height(left_box) + height(right_box))
    )
    spatial_gate = args.base_spatial_gate + args.spatial_gate_per_frame * (gap + 1)
    if normalized_distance > spatial_gate:
        return None

    left_embedding = embeddings.get(left_id, {}).get("end")
    right_embedding = embeddings.get(right_id, {}).get("start")
    if left_embedding is None or right_embedding is None:
        return None
    similarity = float(np.dot(left_embedding, right_embedding))
    if similarity < args.min_similarity:
        return None

    gmc_distance = normalized_distance
    gmc_gate = spatial_gate
    if gmc_transforms is not None:
        matrix = compose_gmc(gmc_transforms, left_frames[-1], right_frames[0])
        point = np.asarray([*center(left_box), 1.0], dtype=np.float64)
        warped = matrix @ point
        warped = warped[:2] / max(float(warped[2]), 1e-9)
        gmc_distance = float(np.linalg.norm(warped - np.asarray(center(right_box)))) / (
            0.5 * (height(left_box) + height(right_box))
        )
        gmc_gate = args.gmc_base_gate + args.gmc_gate_per_frame * (gap + 1)
        if gmc_distance > gmc_gate:
            return None

    cost = (
        args.appearance_weight * (1.0 - similarity)
        + args.spatial_weight * (normalized_distance / spatial_gate)
        + args.gmc_weight * (gmc_distance / gmc_gate) * int(gmc_transforms is not None)
        + args.gap_weight * (gap / max(args.max_gap, 1))
        + args.size_weight * (math.log(size_ratio) / math.log(args.max_size_ratio))
    )
    if cost > args.max_cost:
        return None
    return cost, {
        "from_track": left_id,
        "to_track": right_id,
        "from_frame": left_frames[-1],
        "to_frame": right_frames[0],
        "missing_frames": gap,
        "appearance_similarity": round(similarity, 6),
        "normalized_distance": round(normalized_distance, 6),
        "gmc_normalized_distance": round(gmc_distance, 6),
        "size_ratio": round(size_ratio, 6),
        "cost": round(cost, 6),
    }


def choose_links(
    dataset: dict,
    embeddings: dict,
    gmc_transforms: list[np.ndarray] | None,
    args: argparse.Namespace,
) -> list[dict]:
    track_ids = sorted(
        track_id for track_id, track in dataset["tracks"].items() if track["visible"]
    )
    large = 1e6
    costs = np.full((len(track_ids), len(track_ids)), large, dtype=np.float64)
    details: dict[tuple[int, int], dict] = {}
    for row, left_id in enumerate(track_ids):
        for column, right_id in enumerate(track_ids):
            candidate = candidate_cost(
                left_id, right_id, dataset, embeddings, gmc_transforms, args
            )
            if candidate is None:
                continue
            cost, detail = candidate
            costs[row, column] = cost
            details[(row, column)] = detail
    rows, columns = linear_sum_assignment(costs)
    links = [details[(row, column)] for row, column in zip(rows, columns) if (row, column) in details]
    links.sort(key=lambda item: (item["from_frame"], item["from_track"], item["to_track"]))
    return links


def chains(dataset: dict, links: list[dict]) -> list[list[int]]:
    successor = {row["from_track"]: row["to_track"] for row in links}
    predecessor = {row["to_track"]: row["from_track"] for row in links}
    active = sorted(track_id for track_id, track in dataset["tracks"].items() if track["visible"])
    result = []
    for track_id in active:
        if track_id in predecessor:
            continue
        chain = [track_id]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        result.append(chain)
    return result


def write_cvat(path: Path, dataset: dict, track_chains: list[list[int]]) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "stitched_track_proposals"
    ET.SubElement(task, "size").text = str(dataset["size"])
    ET.SubElement(task, "mode").text = "interpolation"
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(dataset["width"])
    ET.SubElement(original_size, "height").text = str(dataset["height"])

    for output_id, chain in enumerate(track_chains, start=1):
        visible: dict[int, list[float]] = {}
        for source_id in chain:
            for frame, box in dataset["tracks"][source_id]["visible"].items():
                if frame in visible:
                    raise RuntimeError(f"Stitch created overlap at frame {frame}: {chain}")
                visible[frame] = box
        node = ET.SubElement(
            root,
            "track",
            {"id": str(output_id), "label": "person", "source": "auto-stitch"},
        )
        previous_frame = None
        previous_box = None
        for frame in sorted(visible):
            if previous_frame is not None and frame > previous_frame + 1:
                attrs = {
                    "frame": str(previous_frame + 1),
                    "outside": "1",
                    "occluded": "0",
                    "keyframe": "1",
                    "xtl": f"{previous_box[0]:.2f}",
                    "ytl": f"{previous_box[1]:.2f}",
                    "xbr": f"{previous_box[2]:.2f}",
                    "ybr": f"{previous_box[3]:.2f}",
                    "z_order": "0",
                }
                ET.SubElement(node, "box", attrs)
            box = visible[frame]
            attrs = {
                "frame": str(frame),
                "outside": "0",
                "occluded": "0",
                "keyframe": "1",
                "xtl": f"{box[0]:.2f}",
                "ytl": f"{box[1]:.2f}",
                "xbr": f"{box[2]:.2f}",
                "ybr": f"{box[3]:.2f}",
                "z_order": "0",
            }
            ET.SubElement(node, "box", attrs)
            previous_frame, previous_box = frame, box
        if previous_frame is not None and previous_frame + 1 < dataset["size"]:
            attrs = {
                "frame": str(previous_frame + 1),
                "outside": "1",
                "occluded": "0",
                "keyframe": "1",
                "xtl": f"{previous_box[0]:.2f}",
                "ytl": f"{previous_box[1]:.2f}",
                "xbr": f"{previous_box[2]:.2f}",
                "ybr": f"{previous_box[3]:.2f}",
                "z_order": "0",
            }
            ET.SubElement(node, "box", attrs)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = parse_xml(args.annotations)
    embeddings = extract_endpoint_embeddings(
        dataset,
        args.video,
        args.reid_model,
        args.device,
        args.endpoint_samples,
        args.reid_backend,
        args.reid_model_name,
    )
    gmc_transforms = estimate_pairwise_gmc(args.video) if args.use_gmc else None
    links = choose_links(dataset, embeddings, gmc_transforms, args)
    track_chains = chains(dataset, links)
    write_cvat(args.output_dir / "annotations.xml", dataset, track_chains)
    manifest = {
        "source_annotations": str(args.annotations.resolve()),
        "video": str(args.video.resolve()),
        "reid_model": str(args.reid_model.resolve()),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"annotations", "video", "output_dir", "reid_model"}
        },
        "source_tracks": sum(bool(track["visible"]) for track in dataset["tracks"].values()),
        "stitched_tracks": len(track_chains),
        "accepted_links": len(links),
        "links": links,
        "chains": [chain for chain in track_chains if len(chain) > 1],
    }
    (args.output_dir / "stitch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("source_tracks", "stitched_tracks", "accepted_links")}, indent=2))


if __name__ == "__main__":
    main()
