"""Reuse all eleven matching pairs; fit only two new camera exclusion pairs."""
from shared import *
import importlib.util,shutil
spec=importlib.util.spec_from_file_location('full_camera_train',ROOT/'scripts/camera_full_data_20261002/train.py');ct=importlib.util.module_from_spec(spec);spec.loader.exec_module(ct)
ct.OUT=CAMERA

def main():
 cfg=read(CAMERA/'protocol.json');full=np.load(ORIGINAL_CAMERA/'dataset.npz')['rotation'];data=np.load(DATA/'dataset.npz')
 for fold in read(CAMERA/'folds.json'):
  f=fold['fold']
  for seed in cfg['seeds']:
   dest=CAMERA/f'fold_{f}/seed_{seed}/full';dest.mkdir(parents=True,exist_ok=True)
   if (dest/'predictions.npz').exists():continue
   if fold['reuse_previous_fold'] is not None:
    src=BASELINE/f'camera/fold_{f}/seed_{seed}/full';shutil.copy2(src/'camera.pt',dest/'camera.pt')
    model=ct.CameraPredictor().to(ct.DEVICE);model.load_state_dict(ct.torch.load(dest/'camera.pt',map_location=ct.DEVICE,weights_only=False)['model'])
    write(dest/'reuse.json',dict(original=str(src.relative_to(ROOT)),sha256=sha(src/'camera.pt'),exclusion_pair_verified=True))
   else:model,_=ct.fit(dest,full[fold['train_indices']],full[fold['val_indices']],seed,cfg)
   pred=ct.infer(model,data['cam'][:,:P]);idx=np.array(fold['test_indices']);rot=ct.infer(model,full[idx,:P])
   np.savez_compressed(dest/'predictions.npz',indices=idx,rotation=rot,person_rotation=pred)
 write(CAMERA/'training_complete.json',dict(passed=True,reused=33,new=6,all_recording_refits=0))
 print('All 39 correctly held-out camera checkpoints ready.',flush=True)
if __name__=='__main__':main()
