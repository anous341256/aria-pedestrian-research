"""Versioned six-arm experiment; historical inputs and targets are separate."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/biflow_aria_v1_20261002'))
from common import read, write, sha, warp, project_future, rotmat, metrics, P, F, SCALE, gz
import numpy as np
OLD = ROOT/'outputs/biflow_aria_v1_20261002'
PREVIOUS = ROOT/'outputs/factorized_forecast_v1_20261002'
CAMERA = ROOT/'outputs/camera_full_data_20261002'
VERSION = ROOT/'outputs/transformer_uncovered200_20261006'
ARM = 'merged'
OUT = VERSION
BASELINE = ROOT/'outputs/transformer_matched97_20261006/new'
EXPANSION = ROOT/'outputs/sam2_g1_uncovered200_20261006'
DATA = OUT/'data'
ORIGINAL_CAMERA = CAMERA
CAMERA = OUT/'camera'
REPORT = ROOT/'reports/transformer_uncovered200_20261006'
VARIANTS = ['A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry']
NAMES = dict(zip(VARIANTS,['A 先融合','B 仅行人','C 独立分离','D 双向交互','E 单向交互','F 交互后重投影']))


def history(data, idx, stabilized=False, mode='reviewed', rng=None):
    idx=np.asarray(idx); obs=data['xy'][idx,:P].copy(); valid=np.ones(obs.shape[:2],np.float32)
    if mode=='pre_review':
        obs=data['raw'][idx].copy(); valid=data['raw_valid'][idx].copy()
    elif mode in ['controlled_noise','train']:
        if mode=='train':
            choice=rng.random(len(idx)); old=(choice>=.4)&(choice<.6); noisy=choice>=.6
            obs[old]=data['raw'][idx][old]; valid[old]=data['raw_valid'][idx][old]
        else: noisy=np.ones(len(idx),bool)
        jitter=rng.normal(0,.005,obs.shape)*SCALE; obs[noisy]+=jitter[noisy]
        keep=(rng.random(valid.shape)>=.2).astype(np.float32); keep[:,-1]=1; valid[noisy]=keep[noisy]
    origin=obs[:,-1]/SCALE
    points=(warp(obs,data['H'][idx,:P])[0] if stabilized else obs)/SCALE
    rel=points-origin[:,None]; delta=np.zeros_like(rel)
    delta[:,1:]=np.diff(points,axis=1)*(valid[:,1:]*valid[:,:-1])[...,None]
    rel[valid==0]=0
    seq=np.concatenate([rel,delta,valid[...,None],data['seconds'][idx,:P,None]/.5],-1)
    K=data['K'][idx]
    calib=np.stack([K[:,0,0]/SCALE[0],K[:,1,1]/SCALE[1],K[:,0,2]/SCALE[0],K[:,1,2]/SCALE[1]],-1)
    return dict(seq=seq.astype(np.float32),condition=np.concatenate([origin,calib],-1).astype(np.float32),
                origin=origin,input_xy=obs,input_valid=valid)


def target(data,idx,origin,stabilized):
    xy=data['xy'][idx,P:]
    points=warp(xy,data['H'][idx,P:])[0] if stabilized else xy
    return ((points/SCALE-origin[:,None]).reshape(len(idx),-1)).astype(np.float32)


def project(points,rotations,data,idx):
    pairs=[project_future(points[j],rotations[i],data['K'][i]) for j,i in enumerate(idx)]
    return np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])
