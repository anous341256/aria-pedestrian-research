from bulk_common import *
import cv2,torch

def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);inv=read(OUT/'inventory.json');start=time.monotonic();model=torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main','dinov2_vits14',source='local',pretrained=True).eval().cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
 for cl in inv['clips']:
  cid=cl['clip'];dest=OUT/'features'/f'{cid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():continue
  entries=[e for e in inv['entries'] if e['clip']==cid];cap=cv2.VideoCapture(str(ROOT/cl['input']/'rgb.mp4'));assert cap.isOpened();cap.set(cv2.CAP_PROP_POS_FRAMES,cl['lo']);uu=[];ff=[];xx=[];nn=[];gg=[]
  for e in entries:
   ok,im=cap.read();assert ok;crops=[];positions=[]
   for t in e['targets']:
    x1,y1,x2,y2=map(int,t['box']);mask=decode(t['rle'])[y1:y2,x1:x2];crop=im[y1:y2,x1:x2].copy();crop[mask==0]=127;h,w=crop.shape[:2];s=224/max(h,w);ww,hh=max(1,round(w*s)),max(1,round(h*s));x,y=(224-ww)//2,(224-hh)//2;canvas=np.full((224,224,3),127,np.uint8);canvas[y:y+hh,x:x+ww]=cv2.resize(crop,(ww,hh));mc=np.zeros((224,224),np.uint8);mc[y:y+hh,x:x+ww]=cv2.resize(mask,(ww,hh),interpolation=cv2.INTER_NEAREST);occ=mc.reshape(16,14,16,14).mean((1,3));idx=np.flatnonzero(occ.reshape(-1)>=.6)
    if len(idx)>64:idx=idx[np.linspace(0,len(idx)-1,64).round().astype(int)]
    positions.append(idx);crops.extend([cv2.resize(crop,(224,224))[:,:,::-1].copy(),canvas[:,:,::-1].copy()]);uu.append(t['target_key'])
   if crops:
    # Keep original global crop geometry and new local letterbox geometry separate.
    ten=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).cuda().float()/255
    with torch.inference_mode():z=model.forward_features((ten-mean)/std);g=torch.nn.functional.normalize(z['x_norm_clstoken'][::2],dim=-1).cpu().numpy();f=torch.nn.functional.normalize(z['x_norm_patchtokens'][1::2],dim=-1).cpu().numpy()
    gg.extend(g)
    for a,idx in zip(f,positions):
     feat=np.zeros((64,384),np.float32);xy=np.zeros((64,2),np.float32);n=len(idx);feat[:n]=a[idx];xy[:n]=np.column_stack([(idx%16+.5)/16,(idx//16+.5)/16]);ff.append(feat);xx.append(xy);nn.append(n)
  cap.release();np.savez_compressed(dest,uids=np.array(uu),global_features=np.array(gg),features=np.array(ff),xy=np.array(xx),counts=np.array(nn));write(OUT/'feature_progress.json',dict(clip=cid,completed=int(cid),total=len(inv['clips']),seconds=time.monotonic()-start));print('FEATURE',cid,len(uu),flush=True)
 write(OUT/'feature_complete.json',dict(passed=True,clips=len(inv['clips']),targets=sum(len(np.load(OUT/'features'/f'{c["clip"]}.npz')['uids']) for c in inv['clips']),seconds=time.monotonic()-start,weight_sha=sha('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')))
if __name__=='__main__':main()
