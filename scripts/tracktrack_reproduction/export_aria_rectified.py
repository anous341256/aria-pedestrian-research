"""Export Aria RGB with the installed official viewer_mps rectification geometry.

Run with /opt/aria-vrs/bin/python under Ubuntu-24.04. No MPS artifacts are read.
Only RGB samples are requested from the VRS provider. All source samples in the
first 60 device-time seconds are retained, with exact timestamps in a sidecar.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.calibration import DeviceVersion
from projectaria_tools.core.sensor_data import TimeDomain


BASE = Path(__file__).resolve().parents[2]
FFMPEG = Path('/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
GEOMETRY_SOURCE = Path('/opt/aria-vrs/lib/python3.12/site-packages/projectaria_tools/tools/viewer_mps/rerun_viewer_mps.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def windows_path(path):
    path = str(path.resolve())
    if not path.startswith('/mnt/') or path[6:7] != '/':
        raise ValueError(f'Windows FFmpeg requires a /mnt/<drive>/ path: {path}')
    return path[5].upper() + ':\\' + path[7:].replace('/', '\\')


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def calibration_metadata(camera):
    fx, fy = [float(v) for v in camera.get_focal_lengths()]
    cx, cy = [float(v) for v in camera.get_principal_point()]
    return {
        'width_height': [int(v) for v in camera.get_image_size()],
        'projection_params': camera.get_projection_params().tolist(),
        'K': [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        'T_device_camera': camera.get_transform_device_camera().to_matrix().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE / 'data/Aria/raw_vrs/unknown_20260803_164557.vrs')
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/tracktrack_reproduction/aria_164557_first60s_v1/input')
    parser.add_argument('--seconds', type=float, default=60.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'aria_rgb_rectified_first60s.mp4'
    partial_video = args.output / 'aria_rgb_rectified_first60s.partial.mp4'
    manifest_path = args.output / 'manifest.json'
    timestamps_path = args.output / 'frame_timestamps.csv'
    for path in (video, partial_video, manifest_path, timestamps_path):
        if path.exists():
            raise FileExistsError(f'Preserving existing export artifact: {path}')
    source_stat = args.source.stat()
    provider = data_provider.create_vrs_data_provider(str(args.source))
    stream_id = provider.get_stream_id_from_label('camera-rgb')
    device_calibration = provider.get_device_calibration()
    source_camera = device_calibration.get_camera_calib('camera-rgb')
    source_width, source_height = [int(v) for v in source_camera.get_image_size()]
    # Exact installed viewer_mps recipe, without its preview downsampling/JPEG.
    linear_camera = calibration.get_linear_camera_calibration(
        source_width, source_height, source_camera.get_focal_lengths()[0],
        'pinhole', source_camera.get_transform_device_camera(),
    )
    rotate = device_calibration.get_device_version() == DeviceVersion.Gen1
    target_camera = calibration.rotate_camera_calib_cw90deg(linear_camera) if rotate else linear_camera
    width, height = [int(v) for v in target_camera.get_image_size()]
    fps = float(provider.get_image_configuration(stream_id).nominal_rate_hz)
    fps_fraction = Fraction(fps).limit_denominator(100000)
    all_times = np.asarray(provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME), dtype=np.int64)
    first_ns = int(all_times[0])
    end_ns = first_ns + int(args.seconds * 1e9)
    selected_indices = np.flatnonzero((all_times >= first_ns) & (all_times < end_ns))
    selected_times = all_times[selected_indices]
    if not len(selected_indices) or np.any(np.diff(selected_times) <= 0):
        raise ValueError('Missing or non-monotonic RGB timestamps')
    command = [str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', str(fps_fraction), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p', '-threads', '4',
               '-movflags', '+faststart', '-n', windows_path(partial_video)]
    manifest = {
        'status': 'exporting', 'started_utc': utc_now(),
        'source': {'path': windows_path(args.source), 'bytes': source_stat.st_size, 'mtime_ns': source_stat.st_mtime_ns},
        'requested_streams': [{'id': str(stream_id), 'label': 'camera-rgb'}],
        'mps_artifacts_read': [], 'overlays': False,
        'device_generation': str(device_calibration.get_device_version()),
        'projectaria_tools_version': importlib.metadata.version('projectaria-tools'),
        'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
        'geometry': {
            'reference': str(GEOMETRY_SOURCE), 'reference_sha256': sha256(GEOMETRY_SOURCE),
            'reference_function': 'log_mps_to_rerun', 'reference_lines': [847, 890],
            'recipe': 'get_linear_camera_calibration(original width, original height, original fx, pinhole, original T_device_camera); distort_by_calibration(image, linear, original); rotate cw90 only for Gen1',
            'interpolation': 'official distort_by_calibration default BILINEAR',
            'down_sampling_factor': 1, 'rotation_clockwise_degrees': 90 if rotate else 0,
            'source': calibration_metadata(source_camera), 'target': calibration_metadata(target_camera),
            'calibration_usage': 'RGB rectification/export metadata only; not an additional forecasting model input',
        },
        'time': {'crop': '[first RGB device timestamp, first timestamp + 60s)',
                 'first_rgb_device_ns': first_ns, 'end_exclusive_device_ns': end_ns,
                 'nominal_native_fps': fps, 'frames_expected': len(selected_indices),
                 'original_frames_in_recording': len(all_times),
                 'source_frame_indices_zero_based': [int(selected_indices[0]), int(selected_indices[-1])],
                 'actual_last_relative_seconds': float((selected_times[-1]-first_ns)/1e9),
                 'video_timing': 'one-to-one source samples at nominal native CFR; exact jitter retained in frame_timestamps.csv; no frames dropped or duplicated',
                 'min_interval_ms': float(np.min(np.diff(selected_times))/1e6),
                 'median_interval_ms': float(np.median(np.diff(selected_times))/1e6),
                 'max_interval_ms': float(np.max(np.diff(selected_times))/1e6)},
        'video': {'path': windows_path(video), 'width': width, 'height': height,
                  'codec': 'libx264', 'crf': 16, 'pixel_format': 'yuv420p', 'encoding': 'high-quality lossy'},
        'ffmpeg_command': command,
    }
    write_json(manifest_path, manifest)
    print(json.dumps({'output': str(video), 'size': [width, height], 'fps': fps, 'frames': len(selected_indices), 'rotate': rotate}), flush=True)
    started = time.monotonic()
    frames_written = 0
    encoder = None
    try:
        with (args.output / 'ffmpeg_export.log').open('wb') as encoder_log, timestamps_path.open('w', newline='', encoding='utf-8') as time_file:
            writer = csv.writer(time_file)
            writer.writerow(['output_frame_zero_based', 'source_rgb_index_zero_based', 'device_timestamp_ns', 'relative_device_seconds', 'video_cfr_seconds', 'video_minus_device_ms'])
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
            for out_index, source_index in enumerate(selected_indices):
                image_data, image_record = provider.get_image_data_by_index(stream_id, int(source_index))
                image = image_data.to_numpy_array()
                rectified = calibration.distort_by_calibration(image, linear_camera, source_camera)
                if rotate:
                    rectified = np.rot90(rectified, k=3)
                rectified = np.ascontiguousarray(rectified, dtype=np.uint8)
                if rectified.shape != (height, width, 3):
                    raise ValueError(f'Unexpected RGB frame shape: {rectified.shape}')
                encoder.stdin.write(rectified.tobytes())
                device_ns = int(selected_times[out_index])
                relative = (device_ns-first_ns)/1e9
                nominal = out_index/fps
                writer.writerow([out_index, int(source_index), device_ns, f'{relative:.9f}', f'{nominal:.9f}', f'{(nominal-relative)*1000:.6f}'])
                if out_index in (0, len(selected_indices)//2, len(selected_indices)-1):
                    cv2.imwrite(str(args.output / f'preview_{out_index:06d}.png'), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
                frames_written += 1
                if frames_written % 25 == 0 or frames_written == len(selected_indices):
                    time_file.flush()
                    progress = {'status': 'exporting', 'frames_written': frames_written, 'frames_total': len(selected_indices), 'elapsed_seconds': time.monotonic()-started, 'updated_utc': utc_now()}
                    write_json(args.output / 'export_progress.json', progress)
                    print(f'PROGRESS {frames_written}/{len(selected_indices)} frames, {progress["elapsed_seconds"]:.1f}s elapsed', flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=300) != 0:
                raise RuntimeError('FFmpeg failed; inspect ffmpeg_export.log')
        probe_command = [str(FFMPEG.with_name('ffprobe.exe')), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                         '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,pix_fmt:format=duration,size',
                         '-of', 'json', windows_path(partial_video)]
        probe = json.loads(subprocess.check_output(probe_command, timeout=300))
        stream = probe['streams'][0]
        if int(stream['nb_read_frames']) != len(selected_indices) or (stream['width'], stream['height']) != (width, height) or Fraction(stream['avg_frame_rate']) != fps_fraction:
            raise ValueError(f'Encoded video verification failed: {probe}')
        source_after = args.source.stat()
        if (source_after.st_size, source_after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError('Source VRS changed during export')
        partial_video.replace(video)
        manifest.update(status='complete', completed_utc=utc_now(), frames_written=frames_written,
                        elapsed_seconds=time.monotonic()-started, ffprobe=probe)
        manifest['video'].update(bytes=video.stat().st_size, sha256=sha256(video))
        manifest['source']['sha256'] = sha256(args.source)
        write_json(manifest_path, manifest)
        write_json(args.output / 'export_progress.json', {'status': 'complete', 'frames_written': frames_written, 'frames_total': len(selected_indices), 'updated_utc': utc_now()})
        print('COMPLETE ' + json.dumps(manifest['video']), flush=True)
    except BaseException as error:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        manifest.update(status='failed', error=repr(error), frames_written=frames_written, updated_utc=utc_now())
        write_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
