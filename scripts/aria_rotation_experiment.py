"""Controlled Aria Gen2 rotation-compensation experiment.

Run with the E:-hosted WSL /opt/aria-vrs Python. Official projectaria-tools
calibration supplies Fisheye624 project/unproject and T_Device_Camera.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from projectaria_tools.core import data_provider
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from audit_cvat_video_annotations import parse_xml  # noqa: E402
from auto_annotate_people import write_cvat  # noqa: E402


METHODS = (
    "none", "rgb_flow", "legacy_pinhole", "aria_fisheye",
    "aria_plus_rgb_residual", "inverse_rotation", "inverse_plus_rgb_residual",
)
GT_IOU = 0.5
ASSOC_IOU = 0.3
MAX_AGE = 30
FB_ERROR_SMALL_PX = 1.5
SCALE = 0.25


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="Full RGB MP4")
    parser.add_argument("--clip", type=Path, required=True, help="Gold's first-60s MP4")
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    return parser.parse_args()


def read_at(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return frame


def verify_clip(full: Path, clip: Path, frame_count: int) -> list[dict]:
    full_cap = cv2.VideoCapture(str(full))
    clip_cap = cv2.VideoCapture(str(clip))
    if not full_cap.isOpened() or not clip_cap.isOpened():
        raise RuntimeError("Cannot open source or gold clip for frame identity check")
    correlations = []
    for index in range(frame_count):
        ok_full, full_frame = full_cap.read()
        ok_clip, clip_frame = clip_cap.read()
        if not ok_full or not ok_clip:
            raise RuntimeError(f"Source or gold clip ended at frame {index}")
        source = cv2.cvtColor(cv2.resize(full_frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        sample = cv2.cvtColor(cv2.resize(clip_frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        correlation = float(np.corrcoef(sample.ravel(), source.ravel())[0, 1])
        correlations.append(correlation)
        if not np.isfinite(correlation) or correlation < 0.985:
            raise RuntimeError(f"Gold clip differs from source at frame {index}: r={correlation}")
    full_cap.release()
    clip_cap.release()
    return [{"checked_frames": frame_count, "minimum_correlation": min(correlations),
             "median_correlation": float(np.median(correlations))}]


def read_time_map(path: Path, frame_count: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < frame_count:
        raise RuntimeError(f"Time map has {len(rows)} frames, need {frame_count}")
    times = np.asarray([int(row["vrs_device_time_ns"]) for row in rows[:frame_count]], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("First 600 RGB device timestamps are not strictly increasing")
    return times


def load_poses(path: Path, rgb_times: np.ndarray) -> tuple[list[dict | None], list[str]]:
    names = (
        "tracking_timestamp_us", "session_uid", "tx_odometry_device", "ty_odometry_device",
        "tz_odometry_device", "qx_odometry_device", "qy_odometry_device",
        "qz_odometry_device", "qw_odometry_device", "quality_score",
    )
    trajectory = pd.read_csv(path, usecols=list(names))
    times = trajectory["tracking_timestamp_us"].to_numpy(dtype=np.int64) * 1000
    sessions = trajectory["session_uid"].fillna("").to_numpy()
    quality = trajectory["quality_score"].to_numpy(dtype=np.float64)
    translations = trajectory[
        ["tx_odometry_device", "ty_odometry_device", "tz_odometry_device"]
    ].to_numpy(dtype=np.float64)
    quaternions = trajectory[
        ["qx_odometry_device", "qy_odometry_device", "qz_odometry_device", "qw_odometry_device"]
    ].to_numpy(dtype=np.float64)
    poses: list[dict | None] = []
    reasons: list[str] = []
    for query in rgb_times:
        right = int(np.searchsorted(times, query, side="left"))
        if right < len(times) and times[right] == query:
            left = right
        else:
            left = right - 1
        if left < 0 or right >= len(times):
            poses.append(None)
            reasons.append("outside_vio_time_range")
            continue
        if max(query - times[left], times[right] - query) > 5_000_000:
            poses.append(None)
            reasons.append("vio_sample_farther_than_5ms")
            continue
        if sessions[left] != sessions[right] or not sessions[left]:
            poses.append(None)
            reasons.append("vio_session_boundary")
            continue
        if min(quality[left], quality[right]) < 0.5:
            poses.append(None)
            reasons.append("vio_quality_below_0.5")
            continue
        if left == right:
            rotation = Rotation.from_quat(quaternions[left]).as_matrix()
            translation = translations[left]
        else:
            fraction = float((query - times[left]) / (times[right] - times[left]))
            slerp = Slerp(
                [0.0, 1.0], Rotation.from_quat(quaternions[[left, right]])
            )
            rotation = slerp(fraction).as_matrix()
            translation = translations[left] * (1 - fraction) + translations[right] * fraction
        poses.append({"R_OD": rotation, "t_OD": translation, "session": str(sessions[left])})
        reasons.append("valid")
    return poses, reasons


def gold_frames(dataset: dict, count: int) -> list[list[dict]]:
    frames: list[list[dict]] = [[] for _ in range(count)]
    for identity, track in dataset["tracks"].items():
        if track["label"] not in {"", "person"}:
            continue
        for frame, box in track["visible"].items():
            if 0 <= frame < count:
                frames[frame].append({"id": int(identity), "box": np.asarray(box, float), "conf": 1.0})
    for rows in frames:
        rows.sort(key=lambda item: item["id"])
    return frames


def detection_frames(path: Path, count: int) -> list[list[dict]]:
    frames: list[list[dict]] = [[] for _ in range(count)]
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if 0 <= frame < count:
                frames[frame].append(
                    {
                        "id": int(row["detection_id"]),
                        "box": np.asarray([float(row[key]) for key in ("x1", "y1", "x2", "y2")]),
                        "conf": float(row["confidence"]),
                    }
                )
    return frames


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / max(aa + bb - intersection, 1e-9))


def assigned(left: list[np.ndarray], right: list[np.ndarray], threshold: float) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    similarity = np.asarray([[iou(a, b) for b in right] for a in left])
    li, ri = linear_sum_assignment(1.0 - similarity)
    return [(int(i), int(j)) for i, j in zip(li, ri) if similarity[i, j] >= threshold]


def official_warp(camera, points: np.ndarray, relative_rotation: np.ndarray) -> np.ndarray:
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        ray = camera.unproject(np.asarray(point, dtype=np.float64))
        if ray is None:
            continue
        projected = camera.project(relative_rotation @ np.asarray(ray).reshape(3))
        if projected is not None:
            output[index] = np.asarray(projected).reshape(2)
    return output


def pinhole_warp(points: np.ndarray, relative_rotation: np.ndarray) -> np.ndarray:
    f, cx, cy = 2304.0, 1280.0, 960.0
    rays = np.column_stack(((points[:, 0] - cx) / f, (points[:, 1] - cy) / f, np.ones(len(points))))
    rotated = (relative_rotation @ rays.T).T
    output = np.column_stack((f * rotated[:, 0] / rotated[:, 2] + cx,
                              f * rotated[:, 1] / rotated[:, 2] + cy))
    output[rotated[:, 2] <= 0] = np.nan
    return output


def affine_warp(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def warped_points(
    method: str, points: np.ndarray, start: int, end: int, poses: list[dict | None],
    camera, R_DC: np.ndarray, affines: list[np.ndarray | None],
    residual_affines: dict[str, list[np.ndarray | None]],
) -> np.ndarray:
    if method == "none" or start == end:
        return points.copy()
    if method == "rgb_flow":
        result = points.copy()
        for frame in range(start + 1, end + 1):
            if affines[frame] is None:
                return points.copy()
            result = affine_warp(result, affines[frame])
        return result
    if method in residual_affines:
        result = points.copy()
        for frame in range(start + 1, end + 1):
            previous, current = poses[frame - 1], poses[frame]
            correction = residual_affines[method][frame]
            if affines[frame] is None:
                fallback = result.copy()
            else:
                fallback = affine_warp(result, affines[frame])
            if (previous is not None and current is not None
                    and previous["session"] == current["session"]
                    and correction is not None):
                R_device = current["R_OD"].T @ previous["R_OD"]
                R_camera = R_DC.T @ R_device @ R_DC
                if method == "inverse_plus_rgb_residual":
                    R_camera = R_camera.T
                rotated = official_warp(camera, result, R_camera)
                valid = np.isfinite(rotated).all(axis=1)
                if valid.any():
                    fallback[valid] = affine_warp(rotated[valid], correction)
                    result = fallback
                    continue
            # If VIO or projection is unavailable, use RGB GMC for those points.
            result = fallback
        return result
    first, last = poses[start], poses[end]
    if first is None or last is None or first["session"] != last["session"]:
        return points.copy()
    R_device = last["R_OD"].T @ first["R_OD"]
    if method == "legacy_pinhole":
        return pinhole_warp(points, R_device)
    R_camera = R_DC.T @ R_device @ R_DC
    if method == "inverse_rotation":
        R_camera = R_camera.T
    return official_warp(camera, points, R_camera)


def warped_box(method, box, start, end, poses, camera, R_DC, affines, residual_affines):
    x1, y1, x2, y2 = box
    points = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2],
         [(x1+x2)/2, y1], [x2, (y1+y2)/2],
         [(x1+x2)/2, y2], [x1, (y1+y2)/2]],
        dtype=np.float64,
    )
    shifted = warped_points(method, points, start, end, poses, camera, R_DC,
                            affines, residual_affines)
    if not np.isfinite(shifted).all():
        return box.copy()
    return np.asarray([shifted[:, 0].min(), shifted[:, 1].min(),
                       shifted[:, 0].max(), shifted[:, 1].max()])


def flow_matches(frame0: np.ndarray, frame1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray0 = cv2.cvtColor(cv2.resize(frame0, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cv2.resize(frame1, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray0, 800, 0.01, 7)
    if corners is None or len(corners) < 20:
        return np.empty((0, 2)), np.empty((0, 2))
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    target, found, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None, **options)
    if target is None or found is None:
        return np.empty((0, 2)), np.empty((0, 2))
    reverse, back, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, target, None, **options)
    if reverse is None or back is None:
        return np.empty((0, 2)), np.empty((0, 2))
    valid = (found.ravel() == 1) & (back.ravel() == 1)
    valid &= np.linalg.norm(reverse[:, 0, :] - corners[:, 0, :], axis=1) <= FB_ERROR_SMALL_PX
    return corners[valid, 0, :].astype(float) / SCALE, target[valid, 0, :].astype(float) / SCALE


def outside_people(points: np.ndarray, rows: list[dict]) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for row in rows:
        x1, y1, x2, y2 = row["box"]
        margin_x, margin_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        inside = (
            (points[:, 0] >= x1 - margin_x) & (points[:, 0] <= x2 + margin_x)
            & (points[:, 1] >= y1 - margin_y) & (points[:, 1] <= y2 + margin_y)
        )
        mask &= ~inside
    return mask


def geometry_and_rgb_flow(
    clip: Path, count: int, gold: list[list[dict]], poses: list[dict | None],
    camera, R_DC: np.ndarray, out: Path,
) -> tuple[list[np.ndarray | None], dict[str, list[np.ndarray | None]], list[dict]]:
    cap = cv2.VideoCapture(str(clip))
    ok, previous = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read {clip}")
    affines: list[np.ndarray | None] = [None] * count
    residual_affines: dict[str, list[np.ndarray | None]] = {
        "aria_plus_rgb_residual": [None] * count,
        "inverse_plus_rgb_residual": [None] * count,
    }
    rows = []
    for frame in range(1, count):
        ok, current = cap.read()
        if not ok:
            raise RuntimeError(f"Clip ended at frame {frame}")
        first, second = flow_matches(previous, current)
        background = outside_people(first, gold[frame-1]) & outside_people(second, gold[frame])
        # Do not use gold boxes to fit RGB GMC: they are an evaluation-only mask.
        # RANSAC handles moving foreground points without privileged labels.
        fit = np.arange(len(first)) % 2 == 0
        if fit.sum() >= 10:
            affine, _ = cv2.estimateAffinePartial2D(
                first[fit], second[fit], method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
            if affine is not None and np.isfinite(affine).all():
                affines[frame] = affine
        pose0, pose1 = poses[frame-1], poses[frame]
        valid_pose = pose0 is not None and pose1 is not None and pose0["session"] == pose1["session"]
        theta = translation = math.nan
        if valid_pose:
            R_rel = R_DC.T @ pose1["R_OD"].T @ pose0["R_OD"] @ R_DC
            theta = float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()) * 180 / np.pi)
            translation = float(np.linalg.norm(pose1["t_OD"] - pose0["t_OD"]))
            if fit.sum() >= 10:
                for method, rotation in (
                    ("aria_plus_rgb_residual", R_rel),
                    ("inverse_plus_rgb_residual", R_rel.T),
                ):
                    rotated_fit = official_warp(camera, first[fit], rotation)
                    finite = np.isfinite(rotated_fit).all(axis=1)
                    if finite.sum() < 10:
                        continue
                    correction, _ = cv2.estimateAffinePartial2D(
                        rotated_fit[finite], second[fit][finite],
                        method=cv2.RANSAC, ransacReprojThreshold=5.0,
                    )
                    if correction is not None and np.isfinite(correction).all():
                        residual_affines[method][frame] = correction
        heldout = np.arange(len(first)) % 2 == 1
        heldout &= background
        first, second = first[heldout], second[heldout]
        before_calibration_filter = len(first)
        if valid_pose:
            projected = official_warp(camera, first, R_rel)
            calibrated = np.isfinite(projected).all(axis=1)
            first, second = first[calibrated], second[calibrated]
        row = {
            "frame": frame, "pose_valid": int(valid_pose), "rgb_flow_valid": int(affines[frame] is not None),
            "aria_residual_valid": int(residual_affines["aria_plus_rgb_residual"][frame] is not None),
            "inverse_residual_valid": int(residual_affines["inverse_plus_rgb_residual"][frame] is not None),
            "rotation_deg": theta, "translation_m": translation,
            "background_features_before_calibration": before_calibration_filter,
            "background_features": len(first),
        }
        if valid_pose and len(first) >= 20:
            radius = np.linalg.norm((first - np.array([1280.0, 960.0])) / 1600.0, axis=1)
            for method in METHODS:
                warped = warped_points(method, first, frame-1, frame, poses, camera,
                                       R_DC, affines, residual_affines)
                residual = np.linalg.norm(warped - second, axis=1)
                residual = np.where(np.isfinite(residual), residual, np.nan)
                row[f"{method}_median_px"] = float(np.nanmedian(residual))
                row[f"{method}_p90_px"] = float(np.nanpercentile(residual, 90))
                for region, mask in (("center", radius < 0.6), ("edge", radius >= 0.6)):
                    row[f"{method}_{region}_median_px"] = (
                        float(np.nanmedian(residual[mask])) if mask.sum() >= 10 else math.nan
                    )
        rows.append(row)
        previous = current
        if frame % 100 == 0:
            print(f"geometry {frame}/{count-1}", flush=True)
    cap.release()
    fields = list(rows[0])
    for method in METHODS:
        for suffix in ("median_px", "p90_px", "center_median_px", "edge_median_px"):
            key = f"{method}_{suffix}"
            if key not in fields:
                fields.append(key)
    with (out / "frame_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return affines, residual_affines, rows


def detector_to_gold_ids(detected: list[list[dict]], gold: list[list[dict]]) -> list[list[int | None]]:
    mapped = []
    for detections, truth in zip(detected, gold):
        labels: list[int | None] = [None] * len(detections)
        for i, j in assigned(
            [row["box"] for row in detections], [row["box"] for row in truth], GT_IOU
        ):
            labels[i] = truth[j]["id"]
        mapped.append(labels)
    return mapped


def pairwise_scores(
    boxes: list[list[dict]], labels: list[list[int | None]], poses, camera, R_DC,
    affines, residual_affines,
) -> tuple[dict, list[dict]]:
    scores = {}
    details = []
    for method in METHODS:
        tp = fp = fn = pairs = 0
        for frame in range(1, len(boxes)):
            if poses[frame-1] is None or poses[frame] is None:
                continue
            if poses[frame-1]["session"] != poses[frame]["session"]:
                continue
            left = boxes[frame-1]
            right = boxes[frame]
            true_pairs = {
                (i, j) for i, id0 in enumerate(labels[frame-1])
                for j, id1 in enumerate(labels[frame])
                if id0 is not None and id0 == id1
            }
            predicted = set(assigned(
                [warped_box(method, row["box"], frame-1, frame, poses, camera,
                            R_DC, affines, residual_affines)
                 for row in left], [row["box"] for row in right], ASSOC_IOU
            ))
            pair_tp = len(true_pairs & predicted)
            pair_fp = len(predicted - true_pairs)
            pair_fn = len(true_pairs - predicted)
            tp += pair_tp
            fp += pair_fp
            fn += pair_fn
            pairs += 1
            relative = poses[frame]["R_OD"].T @ poses[frame-1]["R_OD"]
            rotation_deg = float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()) * 180 / np.pi)
            translation_m = float(np.linalg.norm(poses[frame]["t_OD"] - poses[frame-1]["t_OD"]))
            scale = float(np.median([math.sqrt(max(0.0, (row["box"][2] - row["box"][0])
                                             * (row["box"][3] - row["box"][1])))
                                     for row in left])) if left else math.nan
            details.append({"method": method, "frame": frame, "tp": pair_tp,
                            "fp": pair_fp, "fn": pair_fn, "rotation_deg": rotation_deg,
                            "translation_m": translation_m, "median_box_scale_px": scale})
        scores[method] = {
            "correct": tp, "wrong": fp, "missed": fn, "frame_pairs": pairs,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2*tp / max(2*tp + fp + fn, 1),
        }
    return scores, details


def grouped_pairwise_scores(details: list[dict]) -> dict:
    base = [row for row in details if row["method"] == "none"]
    if not base:
        return {}
    rotation = np.asarray([row["rotation_deg"] for row in base])
    translation = np.asarray([row["translation_m"] for row in base])
    scale = np.asarray([row["median_box_scale_px"] for row in base])
    rotation_low, rotation_high = np.quantile(rotation, [1/3, 2/3])
    translation_low, translation_high = np.quantile(translation, [1/3, 2/3])
    finite_scale = scale[np.isfinite(scale)]
    scale_cutoff = float(np.median(finite_scale)) if len(finite_scale) else math.nan
    predicates = {
        "all": lambda row: True,
        "high_rotation": lambda row: row["rotation_deg"] >= rotation_high,
        "low_rotation": lambda row: row["rotation_deg"] <= rotation_low,
        "high_translation": lambda row: row["translation_m"] >= translation_high,
        "low_translation": lambda row: row["translation_m"] <= translation_low,
        "small_boxes": lambda row: row["median_box_scale_px"] < scale_cutoff,
        "large_boxes": lambda row: row["median_box_scale_px"] >= scale_cutoff,
    }
    rng = np.random.default_rng(20260924)
    grouped = {}
    for group, include in predicates.items():
        subset = [row for row in details if include(row)]
        block_ids = sorted({row["frame"] // 20 for row in subset})
        if not block_ids:
            continue
        block_counts = {method: {block: np.zeros(3, dtype=int) for block in block_ids}
                        for method in METHODS}
        for row in subset:
            block_counts[row["method"]][row["frame"] // 20] += [row["tp"], row["fp"], row["fn"]]
        group_scores = {}
        resampled_blocks = rng.choice(block_ids, size=(1000, len(block_ids)), replace=True)
        for method in METHODS:
            counts = np.sum([block_counts[method][block] for block in block_ids], axis=0)
            tp, fp, fn = counts.tolist()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            sampled_f1 = []
            sampled_delta = []
            for sample in resampled_blocks:
                sampled = np.sum([block_counts[method][int(block)] for block in sample], axis=0)
                reference = np.sum([block_counts["none"][int(block)] for block in sample], axis=0)
                a, b, c = sampled.tolist()
                rtp, rfp, rfn = reference.tolist()
                value = 2 * a / max(2 * a + b + c, 1)
                baseline = 2 * rtp / max(2 * rtp + rfp + rfn, 1)
                sampled_f1.append(value)
                sampled_delta.append(value - baseline)
            baseline_counts = np.sum([block_counts["none"][block] for block in block_ids], axis=0)
            rtp, rfp, rfn = baseline_counts.tolist()
            baseline_f1 = 2 * rtp / max(2 * rtp + rfp + rfn, 1)
            group_scores[method] = {
                "frame_pairs": sum(1 for row in subset if row["method"] == method),
                "tp": tp, "fp": fp, "fn": fn,
                "f1": f1,
                "f1_block_bootstrap_95ci": np.quantile(sampled_f1, [0.025, 0.975]).tolist(),
                "f1_delta_vs_none": f1 - baseline_f1,
                "delta_block_bootstrap_95ci": np.quantile(sampled_delta, [0.025, 0.975]).tolist(),
            }
        grouped[group] = group_scores
    return {"thresholds": {"rotation_low_deg": float(rotation_low),
                            "rotation_high_deg": float(rotation_high),
                            "translation_low_m": float(translation_low),
                            "translation_high_m": float(translation_high),
                            "median_box_scale_px": scale_cutoff},
            "groups": grouped}


def track_boxes(boxes, method, poses, camera, R_DC, affines,
                residual_affines) -> dict[int, list[dict]]:
    tracks: dict[int, list[dict]] = defaultdict(list)
    active: dict[int, dict] = {}
    next_id = 1
    for frame, detections in enumerate(boxes):
        active = {identity: state for identity, state in active.items()
                  if frame - state["last"] <= MAX_AGE}
        identities = sorted(active)
        predicted = [
            warped_box(method, active[identity]["box"], active[identity]["last"], frame,
                       poses, camera, R_DC, affines, residual_affines)
            for identity in identities
        ]
        matches = assigned(predicted, [row["box"] for row in detections], ASSOC_IOU)
        matched = set()
        for index, det_index in matches:
            identity = identities[index]
            det = detections[det_index]
            active[identity] = {"box": det["box"], "last": frame}
            tracks[identity].append({
                "frame_id": frame, "bbox_xyxy": det["box"].tolist(), "confidence": det["conf"]
            })
            matched.add(det_index)
        for det_index, det in enumerate(detections):
            if det_index in matched:
                continue
            identity = next_id
            next_id += 1
            active[identity] = {"box": det["box"], "last": frame}
            tracks[identity].append({
                "frame_id": frame, "bbox_xyxy": det["box"].tolist(), "confidence": det["conf"]
            })
    return tracks


def write_valid_segment(
    root: Path, inputs: dict[str, list[list[dict]]], gold: list[list[dict]],
    poses: list[dict | None], camera, R_DC: np.ndarray,
    affines: list[np.ndarray | None],
    residual_affines: dict[str, list[np.ndarray | None]], clip: Path,
) -> int:
    """Reset every tracker at the first uninterrupted VIO-valid frame."""
    invalid = [index for index, pose in enumerate(poses) if pose is None]
    start = max(invalid, default=-1) + 1
    if start >= len(poses):
        raise RuntimeError("No uninterrupted VIO-valid suffix is available")
    sessions = {pose["session"] for pose in poses[start:] if pose is not None}
    if len(sessions) != 1:
        raise RuntimeError("VIO session changes inside the candidate valid segment")
    segment = root / "valid_segment"
    segment.mkdir(exist_ok=True)
    gold_tracks: dict[int, list[dict]] = defaultdict(list)
    for frame in range(start, len(gold)):
        for row in gold[frame]:
            gold_tracks[row["id"]].append(
                {"frame_id": frame, "bbox_xyxy": row["box"].tolist(), "confidence": 1.0}
            )
    write_cvat(segment / "gold.xml", gold_tracks, clip, {"processed_frames": len(gold)})
    for input_name, boxes in inputs.items():
        masked_boxes = [[] for _ in range(start)] + boxes[start:]
        input_dir = segment / input_name
        input_dir.mkdir(exist_ok=True)
        for method in METHODS:
            tracks = track_boxes(masked_boxes, method, poses, camera, R_DC,
                                 affines, residual_affines)
            write_cvat(input_dir / f"{method}.xml", tracks, clip,
                       {"processed_frames": len(gold)})
    (segment / "interval.json").write_text(
        json.dumps({"start_frame_inclusive": start, "end_frame_inclusive": len(gold) - 1,
                    "valid_frames": len(gold) - start, "session": next(iter(sessions))}, indent=2),
        encoding="utf-8",
    )
    return start


def bootstrap_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["pose_valid"] and row["background_features"] >= 20]
    if not valid:
        return {"valid_pairs": 0}
    theta = np.asarray([row["rotation_deg"] for row in valid])
    translation = np.asarray([row["translation_m"] for row in valid])
    high_theta = float(np.quantile(theta, 2/3))
    low_translation = float(np.quantile(translation, 1/3))
    subsets = {
        "all": valid,
        "high_rotation_low_translation": [
            row for row in valid if row["rotation_deg"] >= high_theta
            and row["translation_m"] <= low_translation
        ],
        "high_rotation": [row for row in valid if row["rotation_deg"] >= high_theta],
        "low_rotation": [row for row in valid if row["rotation_deg"] < np.quantile(theta, 1/3)],
    }
    rng = np.random.default_rng(20260924)
    result = {
        "valid_pairs": len(valid),
        "rotation_high_tertile_deg": high_theta,
        "translation_low_tertile_m": low_translation,
        "subsets": {},
    }
    for name, subset in subsets.items():
        values = {}
        for method in METHODS:
            differences = np.asarray([
                row["none_median_px"] - row[f"{method}_median_px"] for row in subset
            ])
            differences = differences[np.isfinite(differences)]
            if not len(differences):
                values[method] = {"n": 0}
                continue
            blocks = defaultdict(list)
            for row in subset:
                difference = row["none_median_px"] - row[f"{method}_median_px"]
                if np.isfinite(difference):
                    blocks[int(row["frame"]) // 20].append(float(difference))
            chunks = list(blocks.values())
            bootstrap = []
            for _ in range(1000):
                sample = rng.integers(0, len(chunks), len(chunks))
                bootstrap.append(float(np.median([x for i in sample for x in chunks[i]])))
            values[method] = {
                "n": len(differences),
                "median_residual_reduction_px": float(np.median(differences)),
                "fraction_pairs_improved": float(np.mean(differences > 0)),
                "block_bootstrap_95ci_px": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            }
        result["subsets"][name] = values
    return result


def self_test(camera, R_DC: np.ndarray) -> dict:
    points = np.asarray([[1280., 960.], [1000., 800.], [1500., 1100.], [800., 1400.]])
    identity = official_warp(camera, points, np.eye(3))
    if not np.isfinite(identity).all():
        raise RuntimeError(f"Official projection produced non-finite identity points: {identity}")
    error = float(np.max(np.linalg.norm(identity - points, axis=1)))
    rotation = Rotation.from_euler("xyz", [2, -3, 1], degrees=True).as_matrix()
    forward = official_warp(camera, points, rotation)
    reverse = official_warp(camera, forward, rotation.T)
    if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        raise RuntimeError("Official projection produced non-finite forward/reverse test points")
    inverse_error = float(np.max(np.linalg.norm(reverse - points, axis=1)))
    orthogonality = float(np.linalg.norm(R_DC.T @ R_DC - np.eye(3)))
    if error > 0.01 or inverse_error > 0.01 or orthogonality > 1e-5:
        raise RuntimeError(
            f"Projection self-test failed: identity={error}, inverse={inverse_error}, R={orthogonality}"
        )
    return {"identity_max_error_px": error, "inverse_max_error_px": inverse_error,
            "extrinsic_rotation_orthogonality_error": orthogonality}


def preview(clip, gold, poses, camera, R_DC, affines, residual_affines, out) -> None:
    cap = cv2.VideoCapture(str(clip))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 480))
    if not writer.isOpened():
        cap.release()
        return
    for frame in range(min(180, len(gold))):
        ok, image = cap.read()
        if not ok:
            break
        left = cv2.resize(image, (640, 480))
        middle = left.copy()
        right = left.copy()
        if frame > 0:
            for row in gold[frame-1]:
                box = row["box"]
                raw = (box / 4).astype(int)
                cv2.rectangle(left, tuple(raw[:2]), tuple(raw[2:]), (0, 0, 255), 1)
                if poses[frame-1] is not None and poses[frame] is not None:
                    for method, canvas, color in (
                        ("aria_fisheye", middle, (255, 255, 0)),
                        ("aria_plus_rgb_residual", right, (255, 0, 255)),
                    ):
                        shifted = warped_box(method, box, frame-1, frame,
                                             poses, camera, R_DC, affines,
                                             residual_affines)
                        mapped = (shifted / 4).astype(int)
                        cv2.rectangle(canvas, tuple(mapped[:2]), tuple(mapped[2:]), color, 1)
        for row in gold[frame]:
            box = (row["box"] / 4).astype(int)
            for canvas in (left, middle, right):
                cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)
        for canvas, label in ((left, "No compensation"),
                              (middle, "Aria rotation"),
                              (right, "Aria + RGB residual")):
            cv2.putText(canvas, f"{label} / current GT green", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(np.concatenate([left, middle, right], axis=1))
    writer.release()
    cap.release()


def main() -> None:
    args = args_parser()
    for path in (args.vrs, args.video, args.clip, args.time_map, args.trajectory, args.gold):
        if not path.is_file():
            raise FileNotFoundError(path)
    clip_check = verify_clip(args.video, args.clip, args.frames)
    rgb_times = read_time_map(args.time_map, args.frames)
    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    if provider is None:
        raise RuntimeError(f"Official projectaria-tools could not open {args.vrs}")
    camera = provider.get_device_calibration().get_camera_calib("camera-rgb")
    if camera is None:
        raise RuntimeError("VRS has no factory RGB calibration")
    R_DC = np.asarray(camera.get_transform_device_camera().to_matrix())[:3, :3]
    tests = self_test(camera, R_DC)
    poses, reasons = load_poses(args.trajectory, rgb_times)
    gold = gold_frames(parse_xml(args.gold), args.frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sync_validity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "vrs_device_time_ns", "pose_valid", "reason"))
        writer.writerows((i, int(t), int(p is not None), reasons[i])
                         for i, (t, p) in enumerate(zip(rgb_times, poses)))
    affines, residual_affines, geometry = geometry_and_rgb_flow(
        args.clip, args.frames, gold, poses, camera, R_DC, args.output_dir
    )
    (args.output_dir / "geometry_summary.json").write_text(
        json.dumps(bootstrap_summary(geometry), indent=2), encoding="utf-8"
    )

    inputs = {"oracle_gold_boxes": gold}
    if args.detections:
        inputs["fixed_yolo11l_boxes"] = detection_frames(args.detections, args.frames)
    pairwise = {}
    grouped_pairwise = {}
    pairwise_rows = []
    for input_name, boxes in inputs.items():
        labels = ([[(row["id"]) for row in frame] for frame in gold]
                  if input_name == "oracle_gold_boxes" else detector_to_gold_ids(boxes, gold))
        pairwise[input_name], details = pairwise_scores(
            boxes, labels, poses, camera, R_DC, affines, residual_affines
        )
        grouped_pairwise[input_name] = grouped_pairwise_scores(details)
        pairwise_rows.extend({"input": input_name, **row} for row in details)
        root = args.output_dir / input_name
        root.mkdir(exist_ok=True)
        for method in METHODS:
            tracks = track_boxes(boxes, method, poses, camera, R_DC,
                                 affines, residual_affines)
            write_cvat(root / f"{method}.xml", tracks, args.clip,
                       {"processed_frames": args.frames})
            print(f"{input_name} {method}: tracks={len(tracks)}", flush=True)
    (args.output_dir / "pairwise_summary.json").write_text(
        json.dumps(pairwise, indent=2), encoding="utf-8"
    )
    (args.output_dir / "pairwise_grouped_bootstrap.json").write_text(
        json.dumps(grouped_pairwise, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "pairwise_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    valid_segment_start = write_valid_segment(
        args.output_dir, inputs, gold, poses, camera, R_DC,
        affines, residual_affines, args.clip
    )
    preview(args.clip, gold, poses, camera, R_DC, affines, residual_affines,
            args.output_dir / "rotation_residual_overlay_preview.mp4")
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "vrs": str(args.vrs), "video": str(args.video), "clip": str(args.clip),
                "time_map": str(args.time_map), "trajectory": str(args.trajectory),
                "gold": str(args.gold), "detections": str(args.detections) if args.detections else None,
                "frames": args.frames, "valid_pose_frames": sum(p is not None for p in poses),
                "valid_segment_start": valid_segment_start,
                "self_test": tests, "clip_identity": clip_check,
                "calibration_model": str(camera.get_model_name()),
                "association_iou": ASSOC_IOU, "max_age": MAX_AGE,
                "note": "Aria VIO is a privileged diagnostic reference, not RGB-only inference.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
