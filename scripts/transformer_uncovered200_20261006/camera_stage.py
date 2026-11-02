"""Exactly reuse the 33 cameras matching the frozen eleven held-out pairs."""
from shared import *
from model import CameraPredictor
from train import torch,DEVICE,tensor
import shutil
def main():
 data=dict(np.load(DATA/'dataset.npz'));cfg=read(OUT/'protocol.json')
 for fold in read(OUT/'folds.json'):
  f=fold['fold']
  for seed in cfg['seeds']:
   dest=CAMERA/f'fold_{f}/seed_{seed}/full';dest.mkdir(parents=True,exist_ok=True)
   source=BASELINE/f'camera/fold_{f}/seed_{seed}/full/camera.pt'
   if (dest/'predictions.npz').exists():assert sha(source)==sha(dest/'camera.pt');continue
   shutil.copy2(source,dest/'camera.pt');model=CameraPredictor().to(DEVICE);checkpoint=torch.load(source,map_location=DEVICE,weights_only=False)
   model.load_state_dict(checkpoint['model']);model.eval()
   with torch.no_grad():pred=model(tensor(data['cam'][:,:P])).cpu().numpy()
   np.savez_compressed(dest/'predictions.npz',person_rotation=pred)
   write(dest/'reuse.json',dict(source=str(source.relative_to(ROOT)),sha256=sha(source),training=False))
 write(CAMERA/'training_complete.json',dict(passed=True,reused=33,new=0))
 print(ARM,'33 exact cameras ready',flush=True)
if __name__=='__main__':main()
