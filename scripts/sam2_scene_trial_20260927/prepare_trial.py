"""Independent RGB-only export for the user-authorized SAM2 trial."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/midterm_20260927'))
from common import encoder, finish_encoder, digest, write_json

OUT = ROOT / 'outputs/sam2_scene_trial_20260927/162538_020_040'

def main():
    if (OUT / 'manifest.json').exists():
        print('Complete export exists; reuse', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    vrs = ROOT / 'data/Aria/raw_vrs/unknown_20260803_162538.vrs'
    before = vrs.stat()
    provider = data_provider.create_vrs_data_provider(str(vrs))
    sid = provider.get_stream_id_from_label('camera-rgb')
    source = provider.get_device_calibration().get_camera_calib('camera-rgb')
    w, h = map(int, source.get_image_size())
    target = calibration.get_linear_camera_calibration(w, h, float(source.get_focal_lengths()[0]), 'pinhole', source.get_transform_device_camera())
    fps = float(provider.get_image_configuration(sid).nominal_rate_hz)
    assert fps == 10
    times = provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME)
    indices = list(range(200, 400))
    proc, log = encoder(OUT / 'rgb.mp4', w, h, fps, 18)
    tiles = []
    for f, idx in enumerate(indices):
        img, _ = provider.get_image_data_by_index(sid, idx)
        rgb = calibration.distort_by_calibration(img.to_numpy_array(), target, source)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        proc.stdin.write(bgr.tobytes())
        if f % 10 == 0:
            cv2.imwrite(str(OUT / f'raw_{f:03d}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
            tile = cv2.resize(bgr, (640, 480))
            cv2.rectangle(tile, (0,0), (640,32), (0,0,0), -1)
            cv2.putText(tile, f'162538 | source {idx/10:.1f}s | local f={f}', (8,22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            tiles.append(tile)
            print('EXPORT', f, '/200', flush=True)
    finish_encoder(proc, log)
    for part in range(2):
        group = tiles[part*10:(part+1)*10]
        cv2.imwrite(str(OUT / f'raw_sheet_{part+1}.jpg'), np.vstack([np.hstack(group[j:j+2]) for j in range(0,10,2)]))
    with (OUT / 'frame_timestamps.csv').open('w', newline='', encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['local_frame','source_rgb_index','device_timestamp_ns','relative_recording_s'])
        writer.writerows((f,idx,int(times[idx]),(times[idx]-times[0])/1e9) for f,idx in enumerate(indices))
    assert (before.st_size,before.st_mtime_ns)==(vrs.stat().st_size,vrs.stat().st_mtime_ns)
    write_json(OUT / 'manifest.json', {'recording':vrs.stem,'source_vrs':str(vrs),'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,'source_indices':[200,399],'start_s_nominal':20,'duration_s_nominal':20,'frames':200,'fps':fps,'width':w,'height':h,'source_calibration':source.get_projection_params().tolist(),'target_calibration':target.get_projection_params().tolist(),'T_device_camera':target.get_transform_device_camera().to_matrix().tolist(),'image_domain':'official factory calibration -> linear RGB; not stabilization','inference_inputs':'RGB and explicit object prompts only; no MPS','video_sha256':digest(OUT/'rgb.mp4'),'script_sha256':digest(Path(__file__))})
    print('EXPORT COMPLETE', flush=True)

if __name__ == '__main__':
    main()
