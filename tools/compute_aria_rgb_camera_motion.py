"""Compute RGB-frame-aligned Aria camera/device motion and diagnostic plots.

The parsed Aria export currently contains the RGB MP4-to-VRS time map and the
MPS open-loop device trajectory. If an RGB-camera T_Device_Camera is found in
the exported calibration JSONL, the script computes camera-frame relative
motion. Otherwise it emits a clearly labelled device-frame proxy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSE_COLUMNS = [
    "tx_odometry_device", "ty_odometry_device", "tz_odometry_device",
    "qx_odometry_device", "qy_odometry_device", "qz_odometry_device",
    "qw_odometry_device", "quality_score",
]
RGB_LABELS = {"rgb", "214-1", "rgb-camera", "camera-rgb", "camera"}


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return 0.5 * np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    sine = np.sin(angle)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]) / (2.0 * sine)
    return axis * angle


def nearest_indices(sorted_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_times, query_times, side="left")
    right = np.clip(right, 0, len(sorted_times) - 1)
    left = np.clip(right - 1, 0, len(sorted_times) - 1)
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(
        sorted_times[right] - query_times
    )
    return np.where(choose_left, left, right)


def load_rgb_extrinsic(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Find a likely RGB T_Device_Camera, if the parsed export contains one."""
    if not path.exists():
        return None, None, "device_frame_proxy: calibration file missing"
    candidates: List[Tuple[str, dict]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for camera in record.get("CameraCalibrations", []):
                label = str(camera.get("Label", "")).strip().lower()
                transform = camera.get("T_Device_Camera")
                if transform:
                    candidates.append((label, transform))
    chosen = next((item for item in candidates if item[0] in RGB_LABELS), None)
    if chosen is None:
        return None, None, "device_frame_proxy: no RGB T_Device_Camera in parsed calibration"
    label, transform = chosen
    quat = np.asarray(transform["UnitQuaternion"][1], dtype=np.float64)
    translation = np.asarray(transform["Translation"], dtype=np.float64)
    return quat_xyzw_to_matrix(quat), translation, f"rgb_camera_frame: calibration label={label}"


def pose_matrices(table: pd.DataFrame, rotation: Optional[np.ndarray], translation: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    positions = table[["tx_odometry_device", "ty_odometry_device", "tz_odometry_device"]].to_numpy(float)
    quats = table[["qx_odometry_device", "qy_odometry_device", "qz_odometry_device", "qw_odometry_device"]].to_numpy(float)
    quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    device_rotations = np.stack([quat_xyzw_to_matrix(q) for q in quats], axis=0)
    if rotation is None or translation is None:
        return positions, device_rotations
    # T_W_C = T_W_D * T_D_C, using the Aria calibration naming convention.
    camera_positions = positions + np.einsum("nij,nj->ni", device_rotations, np.broadcast_to(translation, positions.shape))
    camera_rotations = np.einsum("nij,jk->nik", device_rotations, rotation)
    return camera_positions, camera_rotations


def build_motion(recording_dir: Path) -> Tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    map_path = recording_dir / "logs" / "214-1" / "mp4_to_vrs_time_ns.csv"
    traj_path = recording_dir / "csv" / "slam" / "open_loop_trajectory.csv"
    calib_path = recording_dir / "csv" / "slam" / "online_calibration.jsonl"
    rgb_path = recording_dir / "logs" / "214-1" / "video.mp4"
    frame_map = pd.read_csv(map_path).dropna(subset=["mp4_time_ns", "vrs_device_time_ns"])
    frame_map = frame_map.sort_values("mp4_time_ns").reset_index(drop=True)
    frame_map["frame_index"] = np.arange(len(frame_map), dtype=np.int64)
    traj = pd.read_csv(traj_path).dropna(subset=["tracking_timestamp_us", *POSE_COLUMNS])
    traj = traj[traj["quality_score"] > 0].sort_values("tracking_timestamp_us").reset_index(drop=True)
    traj["vrs_device_time_ns"] = (traj["tracking_timestamp_us"].to_numpy(float) * 1000.0).astype(np.int64)
    traj_times = traj["vrs_device_time_ns"].to_numpy(np.int64)
    query_times = frame_map["vrs_device_time_ns"].to_numpy(np.int64)
    match_idx = nearest_indices(traj_times, query_times)
    matched = traj.iloc[match_idx].reset_index(drop=True)
    frame_pose = pd.DataFrame({
        "frame_index": frame_map["frame_index"].to_numpy(np.int64),
        "mp4_time_ns": frame_map["mp4_time_ns"].to_numpy(np.int64),
        "frame_vrs_device_time_ns": query_times,
        "matched_trajectory_time_ns": matched["vrs_device_time_ns"].to_numpy(np.int64),
        "time_error_ns": query_times - matched["vrs_device_time_ns"].to_numpy(np.int64),
    })
    for col in POSE_COLUMNS:
        frame_pose[col] = matched[col].to_numpy()
    frame_pose["sync_valid_5ms"] = frame_pose["time_error_ns"].abs() <= 5e6

    ext_rotation, ext_translation, mode = load_rgb_extrinsic(calib_path)
    world_positions, world_rotations = pose_matrices(frame_pose, ext_rotation, ext_translation)
    rows: List[dict] = []
    nonpositive_dt_count = 0
    for i in range(len(frame_pose) - 1):
        dt = (int(frame_pose.iloc[i + 1].frame_vrs_device_time_ns) - int(frame_pose.iloc[i].frame_vrs_device_time_ns)) / 1e9
        if dt <= 0:
            nonpositive_dt_count += 1
            continue
        relative_rotation = np.dot(world_rotations[i].T, world_rotations[i + 1])
        relative_translation = np.dot(world_rotations[i].T, world_positions[i + 1] - world_positions[i])
        rv = matrix_to_rotvec(relative_rotation)
        rows.append({
            "recording": recording_dir.name,
            "motion_frame": mode.split(":", 1)[0],
            "frame_i": int(frame_pose.iloc[i].frame_index),
            "frame_j": int(frame_pose.iloc[i + 1].frame_index),
            "mp4_time_i_ns": int(frame_pose.iloc[i].mp4_time_ns),
            "mp4_time_j_ns": int(frame_pose.iloc[i + 1].mp4_time_ns),
            "dt_sec": dt,
            "match_error_i_ms": float(frame_pose.iloc[i].time_error_ns / 1e6),
            "match_error_j_ms": float(frame_pose.iloc[i + 1].time_error_ns / 1e6),
            "quality_i": float(frame_pose.iloc[i].quality_score),
            "quality_j": float(frame_pose.iloc[i + 1].quality_score),
            "sync_valid_5ms": bool(frame_pose.iloc[i].sync_valid_5ms and frame_pose.iloc[i + 1].sync_valid_5ms),
            "dx": float(relative_translation[0]), "dy": float(relative_translation[1]), "dz": float(relative_translation[2]),
            "rx": float(rv[0]), "ry": float(rv[1]), "rz": float(rv[2]),
        })
    motion = pd.DataFrame(rows)
    duplicate_times = int(frame_pose["mp4_time_ns"].duplicated().sum())
    quality = {
        "recording": recording_dir.name,
        "rgb_video": str(rgb_path),
        "rgb_frame_count": int(len(frame_pose)),
        "motion_row_count": int(len(motion)),
        "expected_motion_row_count": max(int(len(frame_pose)) - 1, 0),
        "motion_frame": mode,
        "trajectory_rows": int(len(traj)),
        "duplicate_rgb_timestamp_count": duplicate_times,
        "invalid_or_nonpositive_dt_count": nonpositive_dt_count,
        "sync_invalid_frame_count_gt_5ms": int((~frame_pose["sync_valid_5ms"]).sum()),
        "sync_valid_motion_row_count_5ms": int(motion["sync_valid_5ms"].sum()) if len(motion) else 0,
        "sync_invalid_motion_row_count_5ms": int((~motion["sync_valid_5ms"]).sum()) if len(motion) else 0,
        "time_match_abs_error_ms": {
            "mean": float(frame_pose["time_error_ns"].abs().mean() / 1e6),
            "median": float(frame_pose["time_error_ns"].abs().median() / 1e6),
            "p95": float(frame_pose["time_error_ns"].abs().quantile(0.95) / 1e6),
            "max": float(frame_pose["time_error_ns"].abs().max() / 1e6),
        },
        "dt_sec": motion["dt_sec"].describe().to_dict() if len(motion) else {},
        "finite_motion_values": bool(np.isfinite(motion[["dx", "dy", "dz", "rx", "ry", "rz"]].to_numpy()).all()) if len(motion) else False,
        "notes": [
            "VIO/device trajectory is a reference label, not an RGB-only prediction.",
            "If motion_frame starts with device_frame_proxy, RGB-specific extrinsic calibration was not found in parsed exports.",
            "Rolling shutter and scene depth/parallax are not modelled in this first version.",
        ],
    }
    return motion, quality, world_positions, world_rotations, frame_pose["sync_valid_5ms"].to_numpy(bool)


def plot_components(motion: pd.DataFrame, output: Path, mode: str) -> None:
    t = motion["mp4_time_i_ns"].to_numpy(float) / 1e9
    valid = motion["sync_valid_5ms"].to_numpy(bool)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for col, label in zip(["dx", "dy", "dz"], ["dx", "dy", "dz"]):
        axes[0].plot(t, np.where(valid, motion[col], np.nan), label=label, linewidth=1.0)
    for col, label in zip(["rx", "ry", "rz"], ["rx", "ry", "rz"]):
        axes[1].plot(t, np.where(valid, motion[col], np.nan), label=label, linewidth=1.0)
    axes[0].set_ylabel("relative translation (m)")
    axes[1].set_ylabel("rotation vector (rad)")
    axes[1].set_xlabel("RGB video time (s)")
    axes[0].set_title(f"Aria RGB-aligned motion components — {mode} (sync-valid rows only; threshold 5 ms)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, loc="upper right")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_trajectory(positions: np.ndarray, output: Path, recording: str, mode: str) -> None:
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#1f77b4", linewidth=1.4)
    stride = max(len(positions) // 20, 1)
    ax.scatter(positions[::stride, 0], positions[::stride, 1], positions[::stride, 2], s=10, color="#d95f02")
    if len(positions) > 1:
        arrow_idx = np.arange(0, len(positions) - 1, stride)
        delta = positions[arrow_idx + 1] - positions[arrow_idx]
        ax.quiver(positions[arrow_idx, 0], positions[arrow_idx, 1], positions[arrow_idx, 2],
                  delta[:, 0], delta[:, 1], delta[:, 2], color="#555555", alpha=0.65,
                  length=1.0, normalize=False, linewidth=0.7, arrow_length_ratio=0.25)
    ax.scatter(*positions[0], color="#2ca02c", s=45, label="start")
    ax.scatter(*positions[-1], color="#d62728", s=45, label="end")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("world z (m)")
    ax.set_title(f"Aria device/camera world trajectory — {recording}\n{mode}")
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion, quality, positions, _, valid_frame_mask = build_motion(args.recording_dir)
    valid_positions = positions[valid_frame_mask]
    if len(valid_positions) < 2:
        raise RuntimeError("Fewer than two RGB frames have valid trajectory synchronization.")
    motion.to_csv(args.output_dir / "rgb_camera_motion.csv", index=False)
    (args.output_dir / "quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    plot_components(motion, args.output_dir / "motion_components.png", quality["motion_frame"])
    plot_trajectory(valid_positions, args.output_dir / "trajectory_3d.png", args.recording_dir.name, quality["motion_frame"])
    readme = f"""# Aria RGB motion demo\n\nRecording: `{args.recording_dir.name}`\n\nMotion mode: `{quality['motion_frame']}`\n\nThis is an RGB-frame-aligned VIO/device reference. It is not an RGB-only model prediction.\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
