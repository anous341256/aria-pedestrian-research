"""Independent angular scores, split provenance and old runtime replay."""
from shared import *
from run import camera_features,infer
from network import Predictor,torch
from train import DEVICE
import importlib.util
spec=importlib.util.spec_from_file_location('camera_math',ROOT/'scripts/camera_full_data_20261002/common.py');cm=importlib.util.module_from_spec(spec);spec.loader.exec_module(cm)

def main():
 data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json');folds=read(OUT/'folds.json');cf=read(CAMERA/'folds.json');full=np.load(ORIGINAL_CAMERA/'dataset.npz');cfg=read(OUT/'protocol.json');rows=[];replays=[]
 for f,c in zip(folds,cf):
  assert c['test'] not in c['train'] and c['validation'] not in c['train'];idx=np.array(f['test_indices']);ci=np.array(c['test_indices'])
  for seed in cfg['seeds']:
   path=CAMERA/f'fold_{f["fold"]}/seed_{seed}/full';z=np.load(path/'predictions.npz');assert np.array_equal(ci,z['indices']);pred=z['person_rotation'][idx];truth=data['cam'][idx,P:]
   for source,pp,tt in [('person',pred,truth),('full_recording',z['rotation'],full['rotation'][ci,P:])]:
    error=cm.degrees(pp,tt);frozen=cm.degrees(np.zeros_like(pp),tt)
    rows.append(dict(fold=f['fold'],recording=f['test'],seed=seed,source=source,windows=len(pp),mean_angle=float(error.mean()),final_angle=float(error[:,-1].mean()),frozen_mean=float(frozen.mean()),frozen_final=float(frozen[:,-1].mean())))
   if f['fold']<11:
    original=BASELINE/'camera'/f'fold_{f["fold"]}/seed_{seed}/full/camera.pt';assert sha(path/'camera.pt')==sha(original)
    oldids=np.array([i for i in idx if i<199]);features,rot=camera_features(data,f['fold'],seed)
    for variant in VARIANTS:
     op=BASELINE/f'fold_{f["fold"]}/seed_{seed}/{variant}';ck=torch.load(op/'checkpoint.pt',map_location=DEVICE,weights_only=False);model=Predictor(variant,ck['ff_width']).to(DEVICE);model.load_state_dict(ck['model']);model.eval()
     saved=np.load(op/'predictions_reviewed.npz');assert np.array_equal(saved['indices'],oldids);a=infer(model,data,oldids,features,rot);difference=float(np.max(np.abs(a['pred']-saved['pred'])))
     assert difference<.002,(f['fold'],seed,variant,difference)
     replays.append(dict(fold=f['fold'],seed=seed,variant=variant,max_difference_px=difference))
 write(OUT/'camera_audit.json',dict(passed=True,rows=rows,old_replayed=len(replays),old_runtime_replay_max_px=max(r['max_difference_px'] for r in replays),old_replays=replays))
 print('Camera holdouts and 198 previous runtime replays passed.',flush=True)
if __name__=='__main__':main()
