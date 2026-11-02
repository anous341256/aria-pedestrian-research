"""Reconstruct a MOT sequence at its recorded FPS; never alter source data."""
import argparse
import configparser
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ffmpeg', required=True)
    a = ap.parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(a.sequence / 'seqinfo.ini')
    s = cfg['Sequence']
    name, fps, n = s['name'], int(s['frameRate']), int(s['seqLength'])
    w, h = int(s['imWidth']), int(s['imHeight'])
    gtpath = a.sequence / 'gt/gt.txt'
    records = defaultdict(list)
    classes = Counter()
    with gtpath.open(newline='') as f:
        for row in csv.reader(f):
            vals = list(map(float, row))
            classes[int(vals[7])] += 1
            if vals[6] != 0 and int(vals[7]) == 1:
                records[int(vals[0])].append(vals)
    a.output.mkdir(parents=True, exist_ok=True)
    outputs = [a.output / f'{name}_original.mp4', a.output / f'{name}_GT_ID.mp4']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Use a new output directory; refusing to overwrite videos')
    procs = []
    for path, height in zip(outputs, [h, h + 32]):
        cmd = [a.ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{w}x{height}', '-r', str(fps), '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(path)]
        procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE))
    previews = []
    try:
        for frame in range(1, n + 1):
            path = a.sequence / s['imDir'] / f'{frame:06d}{s["imExt"]}'
            img = cv2.imread(str(path))
            if img is None or img.shape[:2] != (h, w):
                raise ValueError(f'Missing/wrong-sized frame: {path}')
            procs[0].stdin.write(img.tobytes())
            overlay = img.copy()
            for row in records[frame]:
                tid = int(row[1])
                x, y, bw, bh = row[2:6]
                # MOT uses 1-based image coordinates; OpenCV uses 0-based coordinates.
                x1, y1 = int(round(x - 1)), int(round(y - 1))
                x2, y2 = int(round(x - 1 + bw)), int(round(y - 1 + bh))
                color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[tid * 37 % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                tx, ty = max(0, min(w - 80, x1)), max(16, min(h - 4, y1 - 3))
                label = f'ID {tid}'
                cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            canvas = cv2.copyMakeBorder(overlay, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            title = f'GROUND TRUTH | {name} | frame {frame}/{n} | {(frame-1)/fps:.2f}s'
            cv2.putText(canvas, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            procs[1].stdin.write(canvas.tobytes())
            if frame in [1, 280, 560, 837]:
                previews.append(canvas)
            if frame % 140 == 0:
                print(f'Rendered {frame}/{n}', flush=True)
    finally:
        for p in procs:
            p.stdin.close()
        statuses = [p.wait() for p in procs]
    if any(statuses):
        raise RuntimeError(f'Encoder failures: {statuses}')
    cv2.imwrite(str(a.output / 'preview_contact_sheet.jpg'), np.vstack([np.hstack(previews[:2]), np.hstack(previews[2:])]))
    verification = []
    for path in outputs:
        cap = cv2.VideoCapture(str(path))
        rate = cap.get(cv2.CAP_PROP_FPS)
        frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        cap.release()
        assert frames == n and abs(rate - fps) < 1e-6, (frames, rate)
        verification.append({'path': str(path), 'decoded_frames': frames, 'fps': rate, 'bytes': path.stat().st_size})
    result = {'source': str(a.sequence), 'fps': fps, 'frames': n, 'duration_seconds': n/fps,
              'source_width': w, 'source_height': h, 'gt_sha256': hashlib.sha256(gtpath.read_bytes()).hexdigest(),
              'filter': 'GT mark != 0 and class == 1; no visibility threshold; no interpolation',
              'all_gt_class_counts': dict(classes), 'rendered_boxes': sum(map(len, records.values())),
              'rendered_ids': len({int(r[1]) for rows in records.values() for r in rows}),
              'output_verification': verification}
    (a.output / 'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
