from pathlib import Path
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/camera_full_data_20261002'
OLD = ROOT / 'outputs/biflow_aria_v1_20261002'
P, F = 6, 15

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def degrees(pred, target):
    # Quaternion geodesic distance, stable near zero and invariant to sign.
    def quaternion(v):
        angle = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.concatenate([v * (.5*np.sinc(angle/(2*np.pi))), np.cos(angle/2)], -1)
    a, b = quaternion(pred.astype(np.float64)), quaternion(target.astype(np.float64))
    dot = np.sum(a*b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    return np.rad2deg(4*np.arctan2(np.linalg.norm(a-b, axis=-1), np.linalg.norm(a+b, axis=-1)))

