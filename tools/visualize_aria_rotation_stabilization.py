"""Visualize Aria RGB video after cumulative VIO rotation compensation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def crop_center(image, fraction):
    height, width = image.shape[:2]
    crop_width = int(width * fraction)
    crop_height = int(height * fraction)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image[top:top + crop_height, left:left + crop_width]


def add_label(image, text, color):
    result = image.copy()
    cv2.rectangle(result, (12, 12), (12 + max(260, len(text) * 16), 50), (0, 0, 0), -1)
    cv2.putText(result, text, (22, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return result


def make_intrinsics(width, height, focal_fraction):
    focal = focal_fraction * max(width, height)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_motion_by_frame(path):
    motion = pd.read_csv(path)
    required = {"frame_j", "rx", "ry", "rz", "sync_valid_5ms"}
    missing = required.difference(motion.columns)
    if missing:
        raise ValueError("Motion CSV missing columns: %s" % sorted(missing))
    return {int(row.frame_j): row for row in motion.itertuples(index=False)}


def build_cumulative_rotations(motion_by_frame, frame_count):
    """C[k] maps coordinates in frame k into the last accumulated reference."""
    cumulative = [np.eye(3, dtype=np.float64) for _ in range(frame_count)]
    valid = np.zeros(frame_count, dtype=bool)
    valid[0] = True
    for frame_index in range(1, frame_count):
        cumulative[frame_index] = cumulative[frame_index - 1].copy()
        row = motion_by_frame.get(frame_index)
        if row is None:
            continue
        # Existing labels are delta = R_previous.T @ R_current, which maps
        # current-frame coordinates back into the previous-frame coordinates.
        # Therefore C_current = C_previous @ delta.
        rotation_vector = np.array([row.rx, row.ry, row.rz], dtype=np.float64).reshape(3, 1)
        delta, _ = cv2.Rodrigues(rotation_vector)
        if bool(row.sync_valid_5ms):
            cumulative[frame_index] = np.dot(cumulative[frame_index - 1], delta)
            valid[frame_index] = True
    return cumulative, valid


def build_smoothed_targets(cumulative, valid, window_frames):
    targets = [np.eye(3, dtype=np.float64) for _ in cumulative]
    for index in range(len(cumulative)):
        if not valid[index]:
            targets[index] = cumulative[index].copy()
            continue
        start = max(0, index - window_frames + 1)
        vectors = []
        for item in cumulative[start:index + 1]:
            vector, _ = cv2.Rodrigues(item)
            vectors.append(vector.reshape(3))
        mean_vector = np.mean(np.asarray(vectors), axis=0).reshape(3, 1)
        targets[index], _ = cv2.Rodrigues(mean_vector)
    return targets


def create_contact_sheet(samples, output_path):
    if not samples:
        return
    tiles = []
    for frame_index, original, stabilized, status in samples:
        original = cv2.resize(original, (480, 360), interpolation=cv2.INTER_AREA)
        stabilized = cv2.resize(stabilized, (480, 360), interpolation=cv2.INTER_AREA)
        left = add_label(original, "original  frame %d" % frame_index, (255, 255, 255))
        right = add_label(stabilized, "compensated  %s" % status, (80, 220, 80) if status == "valid" else (80, 180, 255))
        tiles.append(np.hstack([left, right]))
    rows = []
    for start in range(0, len(tiles), 2):
        row = tiles[start]
        if start + 1 < len(tiles):
            row = np.hstack([row, tiles[start + 1]])
        else:
            row = np.hstack([row, np.zeros_like(row)])
        rows.append(row)
    sheet = np.vstack(rows)
    cv2.imwrite(str(output_path), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-seconds", type=float, default=30.0)
    parser.add_argument("--crop-fraction", type=float, default=0.70)
    parser.add_argument("--focal-fraction", type=float, default=0.90)
    parser.add_argument("--invert-cumulative-motion", action="store_true")
    parser.add_argument("--stabilization-mode", choices=["lock", "smooth"], default="smooth")
    parser.add_argument("--smooth-window-frames", type=int, default=11)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError("Could not open RGB video: %s" % args.video)
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
    source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_count = min(source_frame_count, int(round(args.preview_seconds * fps)))
    motion_by_frame = load_motion_by_frame(args.motion_csv)
    cumulative, sync_valid = build_cumulative_rotations(motion_by_frame, source_frame_count)
    smoothed_targets = build_smoothed_targets(cumulative, sync_valid, max(1, args.smooth_window_frames))
    K = make_intrinsics(width, height, args.focal_fraction)
    K_inv = np.linalg.inv(K)
    crop_width = int(width * args.crop_fraction)
    crop_height = int(height * args.crop_fraction)
    output_width = 960
    output_height = int(round(output_width * crop_height / float(crop_width)))
    writer = cv2.VideoWriter(
        str(args.output_dir / "center_rotation_comparison_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width * 2, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not open output video writer")

    sample_indices = set(np.linspace(0, max(output_count - 1, 0), 6, dtype=int).tolist())
    samples = []
    processed = 0
    compensated_count = 0
    invalid_count = 0
    while processed < output_count:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = processed
        original_crop = crop_center(frame, args.crop_fraction)
        if frame_index == 0 or not sync_valid[frame_index]:
            stabilized = frame
            status = "sync-invalid" if frame_index > 0 else "anchor"
            invalid_count += int(frame_index > 0)
        else:
            if args.stabilization_mode == "smooth":
                motion_matrix = np.dot(smoothed_targets[frame_index].T, cumulative[frame_index])
            else:
                motion_matrix = cumulative[frame_index]
                if args.invert_cumulative_motion:
                    motion_matrix = np.linalg.inv(motion_matrix)
            H = np.dot(np.dot(K, motion_matrix), K_inv)
            stabilized = cv2.warpPerspective(frame, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            status = "valid"
            compensated_count += 1
        stabilized_crop = crop_center(stabilized, args.crop_fraction)
        left = cv2.resize(original_crop, (output_width, output_height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(stabilized_crop, (output_width, output_height), interpolation=cv2.INTER_AREA)
        angle = 0.0
        row = motion_by_frame.get(frame_index)
        if row is not None:
            angle = float(np.linalg.norm([row.rx, row.ry, row.rz]))
        left = add_label(left, "ORIGINAL  t=%.1fs  frame=%d" % (frame_index / fps, frame_index), (255, 255, 255))
        right = add_label(right, "ROTATION COMP  |dtheta|=%.3f rad  %s" % (angle, status), (80, 220, 80) if status == "valid" else (80, 180, 255))
        writer.write(np.hstack([left, right]))
        if frame_index in sample_indices:
            samples.append((frame_index, original_crop, stabilized_crop, status))
        processed += 1
    capture.release()
    writer.release()
    create_contact_sheet(samples, args.output_dir / "center_rotation_comparison_contact.png")
    quality = {
        "source_video": str(args.video),
        "motion_csv": str(args.motion_csv),
        "source_width": width,
        "source_height": height,
        "fps": fps,
        "source_frame_count": source_frame_count,
        "preview_frame_count": processed,
        "preview_seconds": processed / fps,
        "crop_fraction": args.crop_fraction,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "assumed_intrinsics": {
            "model": "pinhole, no distortion",
            "focal_fraction_of_max_dimension": args.focal_fraction,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "invert_cumulative_motion": bool(args.invert_cumulative_motion),
        "stabilization_mode": args.stabilization_mode,
        "smooth_window_frames": int(args.smooth_window_frames),
        "compensated_frame_count": compensated_count,
        "sync_invalid_frame_count_in_preview": invalid_count,
        "sync_valid_motion_rows_available": int(sum(bool(row.sync_valid_5ms) for row in motion_by_frame.values())),
        "notes": [
            "Rotation-only compensation; translation/parallax is intentionally not corrected.",
            "The motion is a VIO/device-frame proxy because RGB T_Device_Camera was not found in parsed exports.",
            "The output video contains video only; source audio is not muxed in this first pass.",
        ],
    }
    (args.output_dir / "stabilization_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    readme = """# Rotation compensation video\n\nLeft: original RGB center crop. Right: cumulative VIO rotation compensation.\n\nThis is a diagnostic visualization, not a metric stabilization benchmark. It uses an approximate pinhole intrinsic matrix and does not correct translation-induced parallax.\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
