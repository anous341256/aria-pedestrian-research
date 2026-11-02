"""Versioned, non-destructive research helpers (not a labeling web app)."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('MIDTERM_OUTPUT_ROOT', str(ROOT / 'outputs/midterm_20260927'))).resolve()
CONFIG = Path(__file__).with_name('config.json')

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def ffmpeg():
    drive = Path('/mnt/e') if os.name != 'nt' else Path('E:/')
    return drive / '下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'

def winpath(p):
    s = str(Path(p).resolve())
    return s[5].upper() + ':\\' + s[7:].replace('/', '\\') if s.startswith('/mnt/') else s

def encoder(path, width, height, fps=10, crf=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    log = path.with_suffix('.ffmpeg.log').open('wb')
    cmd = [str(ffmpeg()), '-hide_banner', '-loglevel', 'error', '-nostdin',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
           '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-threads', '3', '-preset', 'veryfast', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', winpath(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log), log

def finish_encoder(proc, log):
    proc.stdin.close()
    ret = proc.wait(timeout=120)
    log.close()
    if ret:
        raise RuntimeError(f'FFmpeg failed {ret}')

def apply_h(points, H):
    p = np.c_[np.asarray(points).reshape(-1, 2), np.ones(len(points))] @ H.T
    out = np.full((len(p), 2), np.nan)
    valid = np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-8)
    out[valid] = p[valid, :2] / p[valid, 2, None]
    return out

def box_warp(boxes, H):
    b = np.asarray(boxes).reshape(-1,4)
    corners = np.stack([b[:,[0,1]],b[:,[2,1]],b[:,[2,3]],b[:,[0,3]]],axis=1)
    p = apply_h(corners.reshape(-1,2),H).reshape(-1,4,2)
    return np.c_[p.min(axis=1),p.max(axis=1)]

def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2],b[:2]))
    inter = wh.prod()
    return float(inter/(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter+1e-9))
