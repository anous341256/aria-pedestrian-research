"""Independent EDGC/variance arithmetic and all-final-checkpoint measurement replay."""
from pathlib import Path
import sys,importlib.util,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('map_runner',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
# Append scientific tools only after importing the original Torch/Numpy environment.
sys.path.append('/opt/aria-vrs/lib/python3.12/site-packages')
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
OUT=r.OUT;SRC=r.SRC;b=r.b;torch=r.torch;read=r.read;write=r.write
def main():
    d=dict(np.load(SRC/'dataset.npz'));meta=read(SRC/'samples.json');folds=read(SRC/'folds.json')
    feats=read(OUT/'edgc_features.json');rows=read(OUT/'sample_dynamics.json')
    maxfeat={k:0. for k in 'EDGC'};maxmetric=0.;maxprojection=0.;replays=0;prefix=0
    for i,m in enumerate(meta):
        t=(np.array(m['timestamps_ns'][:16],np.int64)-m['timestamps_ns'][15])/1e9
        h=d['boxes'][i,:16,3]-d['boxes'][i,:16,1]
        xy=d['xy'][i,:16];rays=np.concatenate([xy,np.ones((16,1))],axis=1)
        q=np.array([d['H'][i,j]@rays[j] for j in range(16)]);ys=q[:,1]/q[:,2]
        trans=d['translation'][i,:16].astype(float)
        vv=dict(E=float(np.polyfit(t,np.log(h),1)[0]),D=float(np.median(h)/1920),G=float(np.polyfit(t,ys,1)[0]/1920),
                C=float(sum(np.sqrt(sum((trans[j]-trans[j-1])**2)) for j in range(1,16))/(t[-1]-t[0])))
        for k in vv:maxfeat[k]=max(maxfeat[k],abs(vv[k]-feats[i]['features'][k]))
    for f in folds:
      for seed in [17,29,43]:
        cm=r.old.Camera().to(b.DEVICE);cm.load_state_dict(torch.load(SRC/'camera'/f'fold_{f["fold"]}'/f'seed_{seed}'/'checkpoint.pt',map_location=b.DEVICE,weights_only=False)['model']);cm.eval()
        for arm in ['original_long','expanded_long']:
            p=OUT/arm/f'fold_{f["fold"]}'/f'seed_{seed}';z=dict(np.load(p/'curves.npz'));idx=z['indices']
            pm=r.old.Person().to(b.DEVICE);pm.load_state_dict(torch.load(p/'step_240.pt',map_location=b.DEVICE,weights_only=False)['model']);pm.eval()
            pred=r.old.predict(pm,cm,d,meta,idx);truth=b.warp(d['xy'][idx,16:31],d['H'][idx,16:31])[0]
            residual=np.sqrt(np.sum((pred['component']-truth)**2,axis=-1))
            # Independent scipy rotation and explicit ray-to-pixel route.
            rays=np.concatenate([pred['component'],np.ones(pred['component'].shape[:-1]+(1,))],axis=-1)
            ray=np.einsum('nij,ntj->nti',np.linalg.inv(d['K'][idx]),rays)
            rotations=Rotation.from_rotvec(pred['rotation'].reshape(-1,3)).as_matrix().reshape(len(idx),15,3,3)
            projected=np.einsum('nij,ntj->nti',d['K'][idx],np.einsum('ntji,ntj->nti',rotations,ray))
            pixels=projected[...,:2]/projected[...,2:]
            er=np.sqrt(np.sum((pixels-d['xy'][idx,16:31])**2,axis=-1));er[(projected[...,2]<=1e-6)|~np.isfinite(er)]=32000
            maxmetric=max(maxmetric,float(np.max(np.abs(residual-z['residual'][-1]))))
            maxprojection=max(maxprojection,float(np.max(np.abs(er-z['image'][-1]))))
            rr=[row for row in rows if row['arm']==arm and row['fold']==f['fold'] and row['seed']==seed]
            for j,row in enumerate(rr):
                assert row['index']==int(idx[j]);curve=z['residual'][:,j].mean(axis=1)
                manualvar=sum((v-sum(curve[1:])/12)**2 for v in curve[1:])/12
                assert abs(manualvar-row['variance'])<1e-8
                assert abs(curve[-1]-row['final_residual_ADE'])<1e-10
            if f['fold']==0 and seed==17:
                short={k:v[:,:16].copy() if k in ['xy','boxes','R','H','cam','seconds','translation'] else v.copy() for k,v in d.items()}
                a=r.old.predict(pm,cm,short,meta,idx)
                assert all(np.array_equal(a[k],pred[k]) for k in a);prefix+=1
            replays+=1
    assert max(maxfeat.values())<1e-7 and maxmetric<1e-8 and maxprojection<1e-8
    # Pixel-scale sensitivity: supplementary, not a change to maps or training.
    normalized=[]
    for f in folds:
      for seed in [17,29,43]:
        rr=[x for x in rows if x['arm']=='expanded_long' and x['fold']==f['fold'] and x['seed']==seed]
        x=np.array([a['D'] for a in rr]);y=np.array([a['final_residual_ADE'] for a in rr])
        normalized.append(dict(fold=f['fold'],seed=seed,D_pixel_rho=float(spearmanr(x,y).statistic),
                               D_boxheight_normalized_rho=float(spearmanr(x,y/(x*1920)).statistic)))
    hashes=read(OUT/'source_hashes.json');assert all(r.sha(ROOT/p)==h for p,h in hashes.items())
    result=dict(passed=True,final_checkpoint_replays=replays,future_physically_truncated_replays=prefix,
        independent_feature_max_difference=maxfeat,residual_metric_max=maxmetric,scipy_image_metric_max=maxprojection,
        map_row_arithmetic_checks=len(rows),protected_source_files=len(hashes),D_scale_sensitivity=normalized)
    write(OUT/'independent_audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
