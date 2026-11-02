"""Read-only reuse of original geometry and sampling conventions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
LEGACY_CODE = ROOT / 'scripts/biflow_aria_v1_20261002'
sys.path.insert(0, str(LEGACY_CODE))
from common import read, write, sha, warp, project_future, rotmat, metrics, P, F, SCALE, gz
import numpy as np
OLD = ROOT / 'outputs/biflow_aria_v1_20261002'
CAMERA = ROOT / 'outputs/camera_full_data_20261002'
OUT = ROOT / 'outputs/factorized_forecast_v1_20261002'
REPORT = ROOT / 'reports/factorized_forecast_v1_20261002'
VARIANTS = ['separated_dual', 'coupled_dual', 'separated_future']
LABELS = {'separated_dual': '分离双任务', 'coupled_dual': '匹配耦合对照', 'separated_future': '分离只预测'}


def input_history(data, indices, mode='reviewed', rng=None):
    """Only historical slices are returned; targets have a separate function."""
    idx = np.asarray(indices)
    obs = data['xy'][idx, :P].copy()
    valid = np.ones(obs.shape[:2], np.float32)
    if mode == 'pre_review':
        obs = data['raw'][idx].copy(); valid = data['raw_valid'][idx].copy()
    elif mode == 'controlled_noise':
        obs += rng.normal(0, .005, obs.shape) * SCALE
        valid = (rng.random(valid.shape) >= .2).astype(np.float32); valid[:, -1] = 1.
    elif mode == 'train':
        choice = rng.random(len(idx)); old = (choice >= .4) & (choice < .6); noisy = choice >= .6
        obs[old] = data['raw'][idx][old]; valid[old] = data['raw_valid'][idx][old]
        jitter = rng.normal(0, .005, obs.shape) * SCALE; obs[noisy] += jitter[noisy]
        keep = (rng.random(valid.shape) >= .2).astype(np.float32); keep[:, -1] = 1.; valid[noisy] = keep[noisy]
    origin = obs[:, -1] / SCALE
    stable = warp(obs, data['H'][idx, :P])[0] / SCALE
    rel = stable - origin[:, None]
    delta = np.zeros_like(rel)
    delta[:, 1:] = (stable[:, 1:] - stable[:, :-1]) * (valid[:, 1:] * valid[:, :-1])[..., None]
    rel[valid == 0] = 0.
    # Six features, replacing raw width/height with stabilized displacement.
    seq = np.concatenate([rel, delta, valid[..., None], data['seconds'][idx, :P, None] / .5], -1)
    return dict(seq=seq.astype(np.float32), origin=origin.astype(np.float32), input_xy=obs, input_valid=valid)


def targets(data, indices, origin):
    idx = np.asarray(indices)
    stable = warp(data['xy'][idx], data['H'][idx])[0]
    relative = stable / SCALE - origin[:, None]
    return dict(past=relative[:, :P].reshape(len(idx), -1).astype(np.float32),
                future=relative[:, P:].reshape(len(idx), -1).astype(np.float32), stable=stable)


def camera_condition(data, indices, predicted):
    idx = np.asarray(indices)
    return np.concatenate([data['cam'][idx, :P].reshape(len(idx), -1), predicted[idx].reshape(len(idx), -1)], -1).astype(np.float32)


def cameras(fold, seed):
    return dict(full=np.load(CAMERA / f'fold_{fold}/seed_{seed}/full/predictions.npz')['person_rotation'],
                old=np.load(OLD / f'fold_{fold}/seed_{seed}/camera_predictions.npy'))


def project(points, camera, data, indices):
    pairs = [project_future(points[j], camera[i], data['K'][i]) for j, i in enumerate(indices)]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
