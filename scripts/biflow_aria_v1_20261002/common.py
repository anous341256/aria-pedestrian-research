"""Shared IO and geometry for the first Aria dual-task forecasting experiment."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/biflow_aria_v1_20261002'
LABELS=ROOT/'outputs/sam2_near_people_20261002'
VIDEOS=ROOT/'outputs/sam2_g1_v6_v8_20261001'
SCALE=np.array([2560.,1920.]);P=6;F=15
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def center(b):
    b=np.asarray(b);return (b[...,:2]+b[...,2:])/2
def rotvec(R):return np.array([cv2.Rodrigues(r)[0][:,0] for r in np.asarray(R)])
def rotmat(v):return np.array([cv2.Rodrigues(r)[0] for r in np.asarray(v)])
def warp(x,H):
    x=np.asarray(x);h=np.einsum('...ij,...j->...i',H,np.concatenate([x,np.ones(x.shape[:-1]+(1,))],-1));return h[...,:2]/h[...,2:],h[...,2]
def relative(R):return np.einsum('ij,tjk->tik',R[P-1].T,R)
def homographies(R,K):return np.einsum('ij,tjk,kl->til',K,relative(R),np.linalg.inv(K))
def project_future(x,rv,K):
    R=rotmat(rv);Hs=np.einsum('ij,tkj,kl->til',K,R,np.linalg.inv(K));return warp(x,Hs)
def metrics(x,y,depth=None):
    err=np.linalg.norm(x-y,axis=-1);bad=~np.isfinite(err)
    if depth is not None:bad|=depth<=1e-6
    err[bad]=32000.
    return dict(ADE05=float(err[:5].mean()),ADE10=float(err[:10].mean()),ADE15=float(err.mean()),FDE15=float(err[-1]),invalid=int(bad.sum()))
