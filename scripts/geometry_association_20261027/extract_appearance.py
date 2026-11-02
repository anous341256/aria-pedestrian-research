from assoc_common import *
import cv2,time,torch
HUB=Path('/root/.cache/torch/hub/facebookresearch_dinov2_main')
WEIGHT=Path('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
def main():
 torch.set_num_threads(2);inv=read(OUT/'inventory.json');model=torch.hub.load(str(HUB),'dinov2_vits14',source='local',pretrained=True).eval().cuda();start=time.monotonic();alln=0
 mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
 for cl in inv['clips']:
  cid=cl['clip'];dest=OUT/'appearance'/f'{cid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():continue
  entries=[e for e in inv['entries'] if e['clip']==cid];cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));assert cap.isOpened();uids=[];vec=[]
  for e in entries:
   ok,im=cap.read();assert ok;crops=[]
   for t in e['targets']:
    box=np.array(t['box'],int);x1,y1,x2,y2=box;mask=decode(t['rle']);crop=im[y1:y2,x1:x2].copy();crop[mask[y1:y2,x1:x2]==0]=127;crop=cv2.resize(crop,(224,224));crops.append(crop[:,:,::-1].copy());uids.append(t['target_key'])
   if crops:
    ten=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).cuda().float()/255
    with torch.inference_mode():z=model.forward_features((ten-mean)/std)['x_norm_clstoken'];z=torch.nn.functional.normalize(z,dim=-1)
    vec.extend(z.cpu().numpy())
  cap.release();np.savez_compressed(dest,uids=np.array(uids),features=np.array(vec,np.float32));alln+=len(uids);print('APPEARANCE',cid,len(uids),flush=True)
 write(OUT/'appearance_complete.json',dict(passed=True,model='Frozen DINOv2 ViT-S/14 normalized CLS; mask-background-gray crop resized224x224',weight_sha256=sha(WEIGHT),seconds=time.monotonic()-start,targets=sum(len(np.load(OUT/'appearance'/f'{c["clip"]}.npz')['uids']) for c in inv['clips'])))
if __name__=='__main__':main()
