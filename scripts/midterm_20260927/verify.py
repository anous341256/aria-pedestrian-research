"""Read-only artifact verification; writes only verification/inventory JSONs.

Run with /opt/aria-vrs/bin/python. No CVAT calls, no labels changed, no uploads.
"""
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

from common import ROOT, OUT, CONFIG, read_json, write_json, digest, ffmpeg, winpath
from forecast import test_contract


def environment(interpreter, packages):
    code = ("import importlib.metadata as m,json,sys; "
            f"print(json.dumps(dict(python=sys.version,packages={{p:m.version(p) for p in {packages!r}}})))")
    p = subprocess.run([str(interpreter), '-c', code], capture_output=True, text=True, timeout=30)
    if p.returncode:
        return {'error': p.stderr[-1200:], 'returncode': p.returncode}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    out = OUT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    result = {'time_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'scope': 'technical consistency, NOT human identity or geometric annotation accuracy',
              'videos': [], 'clips': []}
    specs = read_json(CONFIG)['clips']
    dev = {s['recording'] for s in specs if s['split'] == 'development'}
    test = {s['recording'] for s in specs if s['split'] == 'test'}
    assert dev.isdisjoint(test)
    result['split'] = {'development': sorted(dev), 'test': sorted(test), 'disjoint': True}
    scripts = Path(__file__).parent
    for p in scripts.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    result['syntax'] = 'all midterm Python scripts parse'
    ids = set()
    count_boxes = count_tracks = count_gaze = count_pairs = 0
    expected_xml = {}
    for ci, s in enumerate(specs):
        clip = OUT / 'clips' / s['id']
        meta = read_json(clip / 'manifest.json')
        vrs = ROOT / 'data/Aria/raw_vrs' / f"{s['recording']}.vrs"
        stat = vrs.stat()
        assert stat.st_size == meta['source_size'] and stat.st_mtime_ns == meta['source_mtime_ns'], vrs
        assert digest(clip / 'rgb.mp4') == meta['video_sha256']
        ref = np.load(clip / 'reference.npz')
        assert ref['R_world_camera'].shape == (100, 3, 3)
        assert np.all(np.diff(ref['times_ns']) > 0)
        assert np.all(np.diff(ref['source_indices']) == 1)
        assert ref['source_indices'][0] == round(s['start_s'] * 10)
        rr = ref['R_world_camera']
        assert np.isfinite(rr).all() and np.max(abs(rr.transpose(0, 2, 1) @ rr - np.eye(3))) < 1e-9
        assert np.max(abs(np.linalg.det(rr) - 1)) < 1e-9
        audit = read_json(clip / 'alignment_audit.json')
        assert all(a['status'] == 'valid' and a['left_gap_ms'] <= 5 and a['right_gap_ms'] <= 5 and a['quality'] >= .5 for a in audit['pose'])
        gaze_samples = len({a['gaze_sample_timestamp_ns'] for a in audit['gaze'] if a['status'] == 'valid_direction'})
        count_gaze += gaze_samples
        trajectory = vrs.parent / f"mps_{s['recording']}_vrs/slam/closed_loop_trajectory.csv"
        graphs = pd.read_csv(trajectory, usecols=['graph_uid'])['graph_uid'].nunique()
        assert graphs == 1, 'Frozen dataset must not cross VIO sessions'
        payload = read_json(clip / 'annotation_v1/tracks.json')
        nboxes = 0
        for t in payload['tracks']:
            global_id = 100 * ci + t['id']
            assert global_id not in ids
            ids.add(global_id)
            rows = t['boxes']
            frames = [r['frame_id'] for r in rows]
            assert len(set(frames)) == len(frames) and min(frames) >= 0 and max(frames) < 100
            assert frames == sorted(frames)
            for r in rows:
                b = np.asarray(r['bbox_xyxy'])
                assert np.isfinite(b).all() and b[0] < b[2] and b[1] < b[3]
                assert min(b) >= -1e-3 and b[2] <= 2560.001 and b[3] <= 1920.001
                assert r['review_status'] == 'assistant_visual_precheck' and r['human_confirmed'] is False
                expected_xml[(global_id, ci * 100 + r['frame_id'])] = b
            nboxes += len(rows)
        count_tracks += len(payload['tracks'])
        count_boxes += nboxes
        geo = read_json(clip / 'geometry/summary.json')
        assert max(geo['self_tests'].values()) < 1e-9
        count_pairs += geo['valid_frame_pairs']
        result['clips'].append({'id': s['id'], 'source_unchanged_size_mtime': True,
                                'video_sha256_matches': True, 'pose_valid': 100,
                                'slam_graphs': int(graphs), 'gaze_unique': gaze_samples,
                                'tracks': len(payload['tracks']), 'boxes': nboxes,
                                'geometry_pairs': geo['valid_frame_pairs'], 'math_tests': geo['self_tests']})
    assert (count_tracks, count_boxes, count_gaze, count_pairs) == (16, 721, 301, 319)
    xml = ET.parse(OUT / 'annotation_v1/annotations.xml').getroot()
    assert int(xml.findtext('meta/task/size')) == 600
    visible = {}
    for t in xml.findall('track'):
        tid = int(t.attrib['id'])
        frames = []
        for b in t.findall('box'):
            f = int(b.attrib['frame'])
            if b.attrib['outside'] == '1':
                continue
            frames.append(f)
            assert b.attrib['keyframe'] == '1'
            visible[(tid, f)] = np.array([float(b.attrib[k]) for k in ['xtl', 'ytl', 'xbr', 'ybr']])
        assert len({f // 100 for f in frames}) == 1
    assert len(xml.findall('track')) == 16 and len(visible) == 721
    assert visible.keys() == expected_xml.keys()
    assert max(float(np.max(abs(visible[k] - b))) for k, b in expected_xml.items()) <= .006
    result['annotation'] = {'tracks': count_tracks, 'visible_boxes': count_boxes, 'xml_json_consistent': True,
                            'cross_clip_id_links': 0, 'human_confirmed': False}
    models_file = np.load(out / 'forecast/models.npz')
    models = {m: {k: models_file[m + '_' + k] for k in ['mean', 'std', 'coef', 'bias']}
              for m in ['ridge_entangled', 'ridge_factorized']}
    result['forecast_contract'] = test_contract(models)
    training = read_json(out / 'forecast/training_windows.json')
    evaluated = read_json(out / 'forecast/windows.json')
    assert len(training) == 22 and len(evaluated) == 66
    assert {r['clip'] for r in training} <= {s['id'] for s in specs if s['split'] == 'development'}
    result['forecast_contract'].update(training_windows=22, evaluation_windows=66,
                                       test_windows=sum(r['split'] == 'test' for r in evaluated))
    sensitivity = read_json(out / 'sensitivity/summary.json')
    assert sensitivity['groups']['all_test']['all']['windows'] == 61
    assert sensitivity['groups']['no_image_edge_contact']['all']['windows'] == 44
    assert sensitivity['groups']['nonoverlap_per_track']['all']['windows'] == 13
    replay = read_json(out / 'sensitivity/rgb_prefix_replay.json')
    assert replay['MPS_inputs'] == [] and replay['future_images_read'] == 0
    assert replay['maximum_transform_cache_difference'] < 1e-8
    assert max(replay['maximum_prediction_cache_differences_px'].values()) < 1e-6
    result['posthoc_audit'] = {'test_windows': 61, 'no_edge_windows': 44,
                               'nonoverlap_per_track_windows': 13, 'rgb_prefix_replay': replay}
    if (OUT / 'annotation_v2/manifest.json').exists():
        review2 = read_json(OUT / 'annotation_v2/manifest.json')
        assert all(digest(Path(p)) == h for p, h in review2['old_assets_sha256'].items())
        confirmed = []
        all_count = 0
        for s in specs:
            old = read_json(OUT / 'clips' / s['id'] / 'annotation_v1/tracks.json')
            new = read_json(OUT / 'annotation_v2/clips' / s['id'] / 'tracks.json')
            assert len(old['tracks']) == len(new['tracks'])
            for a, b in zip(old['tracks'], new['tracks']):
                assert a['track_id'] == b['track_id'] and len(a['boxes']) == len(b['boxes'])
                for x, y in zip(a['boxes'], b['boxes']):
                    assert x['bbox_xyxy'] == y['bbox_xyxy'] and x['frame_id'] == y['frame_id']
                    assert y['human_box_geometry_confirmed'] is False and y['human_confirmed'] is False
                    expected = s['id'] == 'test_150809_080' and b['track_id'] in [1,2] and 0 <= y['frame_id'] <= 53
                    assert y['human_identity_confirmed'] == expected
                    if expected: confirmed.append((s['id'], b['track_id'], y['frame_id']))
                    all_count += 1
        assert len(confirmed) == 108 and all_count == 721
        human_forecast = read_json(out / 'user_identity_confirmed_v2/forecast_windows.json')
        assert len(human_forecast) == 10
        confirmed_set = set(confirmed)
        for r in human_forecast:
            assert all((r['clip'], r['track'], f) in confirmed_set for f in range(r['frame']-9, r['frame']+21))
        result['user_identity_v2'] = {'confirmed_nodes':108, 'unconfirmed_nodes':613,
                                      'geometry_changed':False, 'geometry_confirmed':False,
                                      'outside_scope_upgrades':0, 'forecast_windows':10, 'v1_hashes_unchanged':True}
    # Probe every MP4 and decode first/last frames. Fully decode user-facing bundles and showcases.
    probe = ffmpeg().with_name('ffprobe.exe')
    for p in sorted(OUT.rglob('*.mp4')):
        probe_result = subprocess.run([str(probe), '-v', 'error', '-select_streams', 'v:0',
                                      '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate',
                                      '-of', 'json', winpath(p)], capture_output=True, text=True, timeout=30)
        assert probe_result.returncode == 0, str(p)
        stream = json.loads(probe_result.stdout)['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        cap = cv2.VideoCapture(str(p))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
        assert ok and first.shape[:2] == (int(stream['height']), int(stream['width']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
        ok, last = cap.read()
        assert ok and last.shape == first.shape
        cap.release()
        if p.parent.name == 'annotation_v1':
            assert frames == 600
        elif 'clips' in p.relative_to(OUT).parts:
            assert frames == (99 if p.parent.name == 'geometry' else 100)
        full = p.parent.name in {'showcase', 'annotation_v1'}
        if full:
            proc = subprocess.run([str(ffmpeg()), '-v', 'error', '-nostdin', '-xerror', '-threads', '3',
                                   '-i', winpath(p), '-map', '0:v:0', '-an', '-f', 'null', '-'],
                                  capture_output=True, text=True, timeout=120)
            assert proc.returncode == 0, f'{p}: {proc.stderr}'
        result['videos'].append(dict(path=str(p.relative_to(ROOT)), frames=frames,
                                     first_last_decode=True, full_decode=full, **stream))
        print('VIDEO OK', p.relative_to(OUT), frames, flush=True)
    missing = []
    reports = ROOT / 'reports/midterm_20260927'
    for p in reports.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            dest = (p.parent / link.split('#')[0]).resolve()
            if dest.name in {'verification.json', 'repro_inventory.json'}:
                continue
            if not dest.exists():
                missing.append({'file': p.name, 'target': link})
    assert not missing, missing
    result['report_links'] = 'all local targets exist'
    result['status'] = 'passed'
    result['limitations'] = ['source preservation check uses size/mtime, not original full VRS hash',
                            'video decode and XML consistency do not prove identity correctness',
                            'causal contract test covers code inputs; references and selections remain offline diagnostics']
    write_json(out / 'verification.json', result)
    packages = ['numpy', 'scipy', 'pandas', 'matplotlib', 'opencv-python-headless', 'projectaria-tools']
    inventory = {'time_utc': result['time_utc'], 'platform': platform.platform(), 'output_root': str(OUT),
                 'environments': {'aria-vrs': environment(sys.executable, packages),
                                  'sam2': environment('/opt/cvat-sam2/bin/python', ['torch', 'cvat-sdk', 'sam-2']),
                                  'yolo': environment(ROOT / '.conda-envs/phase1-py310/Scripts/python.exe', ['torch', 'ultralytics', 'numpy'])},
                 'sha256': {}}
    files = list(scripts.glob('*.py')) + list(scripts.glob('*.json')) + list(scripts.glob('reviews/*.json')) + list(reports.glob('*.md'))
    files += [ROOT / 'tracking/botsort_reid.yaml', ROOT / 'tracking/auto_annotate_people.py',
              ROOT / 'scripts/aria_rectified_rotation_background.py', ROOT / 'models/tracking/yolo11l.pt',
              ROOT / 'tools/cvat-models-official/tracker/sam2/func.py']
    cache = Path('/opt/cvat-sam2-cache/huggingface/hub/models--facebook--sam2.1-hiera-large/snapshots')
    files += list(cache.glob('*/*.pt'))
    assert all('error' not in env for env in inventory['environments'].values()), inventory['environments']
    for p in files:
        inventory['sha256'][str(p)] = digest(p)
    inventory['verification_note'] = 'Source/manifests retain execution-time hashes; final source adds graph audit and configurable output root, which do not change computed arrays.'
    write_json(out / 'repro_inventory.json', inventory)
    print('PASS', len(result['videos']), 'videos;', count_tracks, 'tracks;', count_boxes, 'boxes;', count_pairs, 'geometry pairs', flush=True)


if __name__ == '__main__':
    main()
