"""Independent SciPy rotation and pinhole projection audit, without torch."""
from prepare import *
from scipy.spatial.transform import Rotation
def main():
 d=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cfg=read(OUT/'protocol.json');folds=read(OUT/'folds.json');cfolds=read(OUT/'camera_folds.json');camera=np.load(OUT/'camera_dataset.npz');names=read(CAM/'inventory.json')['record_names'];maxgeo=0.;maxangle=0.;maxmetric=0.;maxref=0.;count=0
 for m in meta:
  ns=np.asarray(m['timestamps_ns']);assert (ns[:16]<=m['timestamp_ns']).all() and (ns[16:]>m['timestamp_ns']).all()
 relative=np.swapaxes(d['R'][:,15],-1,-2)[:,None]@d['R'];sr=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(-1,21,3);maxref=float(np.max(np.abs(sr-d['cam'])));assert maxref<1e-7
 matrices=d['K'][:,None]@relative@np.linalg.inv(d['K'])[:,None];assert np.max(np.abs(matrices-d['H']))<1e-7
 for fold,cf in zip(folds,cfolds):
  sets=[set(camera['record_index'][cf[k]].tolist()) for k in ['train_indices','val_indices','test_indices']];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);assert names[next(iter(sets[1]))]==fold['validation'] and names[next(iter(sets[2]))]==fold['test']
  idx=np.asarray(fold['test_indices'])
  for seed in cfg['seeds']:
   for arm in cfg['arms']:
    p=np.load(OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/person/predictions.npz');assert np.array_equal(p['indices'],idx)
    direction=np.einsum('nij,ntj->nti',np.linalg.inv(d['K'][idx]),np.concatenate([p['component'],np.ones((len(idx),5,1))],-1));r=Rotation.from_rotvec(p['rotation'].astype(float).reshape(-1,3)).as_matrix().reshape(-1,5,3,3);ray=np.einsum('ntji,ntj->nti',r,direction);h=np.einsum('nij,ntj->nti',d['K'][idx],ray);xy=h[...,:2]/h[...,2:]
    maxgeo=max(maxgeo,float(np.max(np.abs(xy-p['image']))));err=np.linalg.norm(xy-d['xy'][idx,16:21],axis=-1);err[(h[...,2]<=1e-6)|~np.isfinite(err)]=32000.;maxmetric=max(maxmetric,float(np.max(np.abs(err-p['image_error']))))
    a=Rotation.from_rotvec(p['rotation'].astype(float).reshape(-1,3));b=Rotation.from_rotvec(d['cam'][idx,16:21].reshape(-1,3));angle=np.rad2deg((a.inv()*b).magnitude()).reshape(-1,5);maxangle=max(maxangle,float(np.max(np.abs(angle-p['angle_deg']))));count+=len(idx)*5
 assert maxgeo<1e-6 and maxmetric<1e-6 and maxangle<1e-5
 check=dict(passed=True,evaluated_future_positions=count,independent_projection_max_px=maxgeo,independent_error_max_px=maxmetric,independent_SO3_angle_max_deg=maxangle,reference_rotation_max_rad=maxref,all_recording_splits_disjoint=True,historical_timestamps_only=True)
 write(OUT/'independent_audit.json',check);print(check)
if __name__=='__main__':main()
