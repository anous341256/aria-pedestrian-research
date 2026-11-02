"""Frozen inputs and mask IO for the SAM3/SAM2 five-person comparison."""
from pathlib import Path
import gzip
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/sam3_vs_sam2_20261001'
VIDEO = ROOT / 'outputs/sam2_fullvideos_20260927/201129/rgb.mp4'
SEEDS = ROOT / 'outputs/sam2_five_people_20261001/initial_masks.json'
START, STOP, FPS, W, H = 100, 340, 10, 2560, 1920
IDS = list('ABCDE')
COLORS = dict(zip(IDS, [(255,210,0),(0,220,255),(75,230,100),(220,95,235),(0,130,255)]))
sys.path.insert(0, str(ROOT / 'scripts/midterm_20260927'))
from common import read_json, write_json, digest, encoder, finish_encoder

BASELINES = [
    ('sam2_native', 'sam2_five_people_20261001', 'variant', 'baseline', 'RGB + initial masks'),
    ('sam2_negative_memory', 'sam2_five_people_20261001', 'variant', 'negative_memory', 'RGB + initial masks + later reviewed negative prompts'),
    ('sam2_motion_no_camera', 'sam2_motion_tracker_v1_20261001', 'branch', 'motion_no_camera', 'RGB + initial masks + motion/identity rules'),
    ('sam2_motion_camera', 'sam2_motion_tracker_v1_20261001', 'branch', 'motion_with_camera', 'RGB + initial masks + offline MPS rotation + rules'),
    ('sam2_motion_weight10_camera', 'sam2_motion_weight10_20261001', 'branch', 'motion_with_camera', 'RGB + initial masks + offline MPS rotation + rules; motion weight 10%'),
    ('sam2_reliability_camera', 'sam2_reliability_prior_20261001', 'branch', 'motion_with_camera', 'RGB + initial masks + offline MPS rotation + reliability rules'),
]


def decode(rle):
    mask = np.zeros((H, W), dtype=bool)
    if rle is None:
        return mask
    counts = np.asarray(rle[:-4], dtype=np.int64)
    x0, y0, x1, y1 = map(int, rle[-4:])
    if np.any(counts < 0) or not (0 <= x0 <= x1 < W and 0 <= y0 <= y1 < H):
        raise ValueError('Invalid CVAT mask RLE')
    flat = np.repeat(np.arange(len(counts)) % 2 == 1, counts)
    mask[y0:y1+1, x0:x1+1] = flat.reshape(y1-y0+1, x1-x0+1)
    return mask


def encode(mask):
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.where(mask)
    if not len(xx):
        return [1, 0, 0, 0, 0]
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    flat = mask[y0:y1+1, x0:x1+1].ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, len(flat)]
    lengths = np.diff(boundaries).tolist()
    if flat[0]:
        lengths.insert(0, 0)
    return lengths + [x0, y0, x1, y1]


def mask_area(rle):
    return int(sum(rle[:-4][1::2])) if rle else 0


def archive(path, field=None, value=None):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if field and row[field] != value:
                continue
            key = (row['frame'], row['id'])
            if key in result:
                raise ValueError(f'Duplicate mask: {key}')
            result[key] = row['rle']
    return result


def panel(bgr, masks, title, frame, width=640):
    height = width * 3 // 4
    im = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    small={identity:cv2.resize(decode(rle).astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)
           for identity,rle in masks.items()}
    counts=sum(small.values()) if small else np.zeros((height,width),dtype=np.uint8)
    for identity, mask in small.items():
        hit = mask > 0
        unique=hit & (counts==1)
        im[unique] = (.65 * im[unique] + .35 * np.array(COLORS[identity])).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, contours, -1, COLORS[identity], 1)
        if hit.any():
            _, _, _, (x, y) = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 5))
            dx={'A':-22,'B':25,'C':13,'D':-18,'E':19}[identity]
            label_x=max(5,min(width-18,x+dx));label_y=max(19,min(height-4,y-15))
            cv2.line(im,(x,y),(label_x+4,label_y-4),COLORS[identity],1,cv2.LINE_AA)
            cv2.putText(im, identity, (label_x,label_y), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(im, identity, (label_x,label_y), cv2.FONT_HERSHEY_SIMPLEX, .6, COLORS[identity], 2, cv2.LINE_AA)
    yy,xx=np.indices((height,width))
    im[(counts>1)&((xx+yy)%12<3)]=235
    header = np.full((56, width, 3), 22, np.uint8)
    cv2.putText(header, title, (10,22), cv2.FONT_HERSHEY_SIMPLEX, .53, (245,245,245), 1, cv2.LINE_AA)
    cv2.putText(header, f'Source {frame / FPS:.1f}s | frame {frame} | fixed IDs A-E', (10,45), cv2.FONT_HERSHEY_SIMPLEX, .43, (200,200,200), 1, cv2.LINE_AA)
    return np.vstack([header, im])
