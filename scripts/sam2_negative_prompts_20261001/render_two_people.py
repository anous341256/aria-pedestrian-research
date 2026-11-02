"""Show archived A and each inferred B together without changing annotation data."""
import gzip
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from cvat_sdk.masks import decode_mask

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/midterm_20260927'))
from common import encoder, finish_encoder, read_json, write_json, digest

SOURCE = ROOT / 'outputs/sam2_negative_prompts_20261001'
DEST = SOURCE / 'two_people_view'
W, H = 2560, 1920
A_COLOR, B_COLOR, BOTH_COLOR = (255, 210, 0), (0, 220, 255), (230, 65, 230)


def load_rows(path, frame_key, identity_key):
    rows = {}
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row[frame_key], row[identity_key])] = row['rle']
    return rows


def decode(rle, size):
    if rle is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    full = decode_mask(rle, image_width=W, image_height=H)
    return cv2.resize(full.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def text(image, value, origin, color=(238, 238, 238), scale=.65):
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def marker(image, mask, identity, color, side):
    if not mask.any():
        return
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, _, _, (x, y) = cv2.minMaxLoc(distance)
    label_x = max(5, min(image.shape[1]-100, x + side*75))
    label_y = max(25, y-35 if side < 0 else y+30)
    label_y = min(image.shape[0]-8, label_y)
    cv2.line(image, (x, y), (label_x+12, label_y-8), color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (label_x-3, label_y-23), (label_x+27, label_y+5), (15, 15, 15), -1)
    text(image, identity, (label_x, label_y), color, .85)


def panel(rgb, a, b, title, timestamp, case, points=None):
    rgb = rgb.copy()
    overlap = a & b
    for region, color in [(a & ~b, A_COLOR), (b & ~a, B_COLOR), (overlap, BOTH_COLOR)]:
        rgb[region] = (.56*rgb[region] + .44*np.array(color)).astype(np.uint8)
    for mask, color in [(a, A_COLOR), (b, B_COLOR)]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color, 2)
    marker(rgb, a, 'A', A_COLOR, -1)
    marker(rgb, b, 'B', B_COLOR, 1)
    if points:
        for x, y in points:
            pos = (round(x*rgb.shape[1]/W), round(y*rgb.shape[0]/H))
            cv2.circle(rgb, pos, 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(rgb, pos, 4, (0, 0, 255), -1, cv2.LINE_AA)
    canvas = np.full((rgb.shape[0]+128, rgb.shape[1], 3), (24, 24, 24), dtype=np.uint8)
    canvas[48:48+rgb.shape[0]] = rgb
    text(canvas, title, (14, 30), scale=.7 if rgb.shape[1] >= 900 else .54)
    base = 48+rgb.shape[0]
    text(canvas, f'Source {timestamp:.1f}s | A: '+('mask' if a.any() else 'no mask')+
         ' | B: '+('mask' if b.any() else 'no mask'), (14, base+25), scale=.57)
    text(canvas, f'A = archived S{case["donor_id"]}', (14, base+49), A_COLOR, .56)
    text(canvas, f'B = tracked S{case["target_id"]}', (240, base+49), B_COLOR, .56)
    text(canvas, 'Purple = overlap', (470, base+49), BOTH_COLOR, .52)
    if points:
        text(canvas, 'RED: negative points for B', (14, base+72), (75, 100, 255), .54)
    else:
        text(canvas, 'No mask does not establish absence.', (14, base+72), (180, 180, 180), .5)
    return canvas


def run():
    DEST.mkdir(exist_ok=False)
    protocol = read_json(SOURCE / 'protocol.json')
    manifest = dict(purpose='Display both people using existing masks; no model inference or annotation changes.',
                    colors=dict(A='cyan', B='yellow', overlap='purple', negative_points='red'),
                    A_scope='Archived automatic masks. Only selected negative-point locations were reviewed; '
                            'not a fully verified donor track.', cases=[])
    for case in protocol['cases']:
        name = case['name']
        dest = DEST / name
        dest.mkdir()
        rgb_path = ROOT / case['clip'] / 'rgb.mp4'
        donor_path = ROOT / case['clip'] / 'sam2_raw_v1/masks_rle.jsonl.gz'
        b_path = SOURCE / name / 'masks_rle.jsonl.gz'
        pure_path = SOURCE / 'negative_only_ablation' / name / 'masks_rle.jsonl.gz'
        donors = load_rows(donor_path, 'frame', 'id')
        results = load_rows(b_path, 'frame', 'variant')
        pure = load_rows(pure_path, 'frame', 'variant')
        selected = ([10, 100, 105, 111, 120, 139] if name == 'food_overlap'
                    else [100, 160, 170, 190, 191, 193, 270, 330, 339])
        movie = dest / 'A_B_comparison.mp4'
        proc, log = encoder(movie, 1920, 848, fps=10)
        cap = cv2.VideoCapture(str(rgb_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, case['start'])
        both_count, output_count = 0, 0
        for f in range(case['start'], case['stop']):
            ok, frame = cap.read()
            assert ok
            size = (960, 720)
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            a_rle = donors.get((f, case['donor_id']))
            a = decode(a_rle, size)
            baseline = decode(results[(f, 'baseline')], size)
            corrected = decode(results[(f, 'negative_points')], size)
            points = case['negative_prompts'].get(str(f), {}).get('points')
            timestamp = (f+case['source_offset'])/10
            left = panel(small, a, baseline, 'Original propagation | A + B', timestamp, case)
            right = panel(small, a, corrected, 'Negative points + memory refresh | A + B', timestamp, case, points)
            canvas = np.hstack([left, right])
            canvas[:, 958:962] = 240
            proc.stdin.write(canvas.tobytes())
            output_count += 1
            both_count += int(a.any() and corrected.any())
            if f in selected:
                cv2.imwrite(str(dest / f'A_B_{f:04d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
                size4 = (640, 480)
                original4 = cv2.resize(frame, size4, interpolation=cv2.INTER_AREA)
                a4 = decode(a_rle, size4)
                conditions = [
                    ('Original propagation', results[(f, 'baseline')], None),
                    ('Negative points only', pure[(f, 'negative_points')], points),
                    ('Memory refresh only', results[(f, 'memory_refresh')], None),
                    ('Negative points + memory refresh', results[(f, 'negative_points')], points)]
                tiles = [panel(original4, a4, decode(rle, size4), title, timestamp, case, pts)
                         for title, rle, pts in conditions]
                four = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
                cv2.imwrite(str(dest / f'four_conditions_A_B_{f:04d}.jpg'), four,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            if (f-case['start']+1) % 50 == 0:
                print(f'{name}: rendered {f-case["start"]+1}/{case["stop"]-case["start"]}', flush=True)
        cap.release()
        finish_encoder(proc, log)
        verify = cv2.VideoCapture(str(movie))
        decoded = 0
        while True:
            ok, image = verify.read()
            if not ok:
                break
            assert image.shape[:2] == (848, 1920)
            decoded += 1
        verify.release()
        assert decoded == output_count == case['stop']-case['start']
        inputs = {key:dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for key, path in
                  [('rgb', rgb_path), ('archived_A', donor_path), ('B_results', b_path), ('pure_negative_B', pure_path)]}
        assert inputs['rgb']['sha256'] == case['source_hashes']['rgb']
        assert inputs['archived_A']['sha256'] == case['source_hashes']['masks']
        manifest['cases'].append(dict(case=name, start=case['start'], stop_exclusive=case['stop'],
                                     fps=10, decoded_frames=decoded, video=str(movie.relative_to(ROOT)),
                                     frames_with_both_masks_in_corrected_panel=both_count, inputs=inputs))
        print(f'{name}: full decode passed, {decoded} frames; both masks in {both_count} frames', flush=True)
    manifest['verification'] = 'All output videos decoded completely; source RGB and archived masks unchanged.'
    write_json(DEST / 'manifest.json', manifest)


if __name__ == '__main__':
    run()
