"""Read original Aria VRS files with Meta's projectaria-tools.

This creates a compact inventory and factory RGB calibration reference without
re-encoding media or modifying the original recordings.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain


STREAMS_TO_CHECK = ("214-1", "1201-1", "1201-2", "1201-3", "1201-4")


def inspect_vrs(vrs_path: Path, parsed_root: Path) -> dict:
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError("projectaria-tools could not open this VRS")

    streams = []
    for stream_id in provider.get_all_streams():
        stream = {
            "id": str(stream_id),
            "label": provider.get_label_from_stream_id(stream_id),
            "data_records": int(provider.get_num_data(stream_id)),
        }
        try:
            stream["first_device_time_ns"] = int(
                provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
            )
            stream["last_device_time_ns"] = int(
                provider.get_last_time_ns(stream_id, TimeDomain.DEVICE_TIME)
            )
        except Exception as exc:
            stream["device_time_error"] = str(exc)
        streams.append(stream)

    calibration = provider.get_device_calibration()
    rgb_calibration = None
    if calibration is not None:
        camera = calibration.get_camera_calib("camera-rgb")
        if camera is not None:
            readout_time = camera.get_readout_time_sec()
            rgb_calibration = {
                "model": str(camera.get_model_name()),
                "image_size": [int(value) for value in camera.get_image_size()],
                "projection_parameters": camera.get_projection_params().tolist(),
                "transform_device_camera": camera.get_transform_device_camera()
                .to_matrix()
                .tolist(),
                "readout_time_sec": float(readout_time) if readout_time is not None else None,
            }

    recording_out = parsed_root / vrs_path.stem
    media = {}
    for stream_id in STREAMS_TO_CHECK:
        candidates = (
            recording_out / "logs" / stream_id / "video.mp4",
            recording_out / "mp4" / f"{vrs_path.stem}_{stream_id}.mp4",
        )
        existing = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
        media[stream_id] = {
            "exists": existing is not None,
            "path": str(existing) if existing else None,
            "bytes": existing.stat().st_size if existing else 0,
        }

    csv_dir = recording_out / "csv"
    return {
        "recording": vrs_path.stem,
        "vrs_path": str(vrs_path),
        "vrs_bytes": vrs_path.stat().st_size,
        "device_version": str(provider.get_device_version()),
        "streams": streams,
        "factory_rgb_calibration": rgb_calibration,
        "existing_media": media,
        "existing_gen2_csv": {
            "marker": (csv_dir / ".gen2_csv_done").is_file(),
            "open_loop_trajectory": (csv_dir / "slam" / "open_loop_trajectory.csv").is_file(),
            "online_calibration": (csv_dir / "slam" / "online_calibration.jsonl").is_file(),
            "eye_gaze": (csv_dir / "eye_gaze" / "generalized_eye_gaze.csv").is_file(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aria-root", type=Path, default=Path("/mnt/e/Aria"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.aria_root
    output = args.output or root / "parsed" / "vrs_inventory.json"
    vrs_files = sorted(root.glob("*.vrs"))
    if not vrs_files:
        parser.error(f"No VRS files found in {root}")

    recordings = []
    failures = []
    for index, vrs_path in enumerate(vrs_files, start=1):
        print(f"[{index}/{len(vrs_files)}] {vrs_path.name}", flush=True)
        try:
            recordings.append(inspect_vrs(vrs_path, root / "parsed"))
        except Exception as exc:
            failures.append({"vrs_path": str(vrs_path), "error": str(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "tool": "projectaria-tools",
                "tool_version": version("projectaria-tools"),
                "recordings": recordings,
                "failures": failures,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(f"Inventory: {output}")
    print(f"Opened: {len(recordings)}/{len(vrs_files)}; failures: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
