from partial_common import *
import cv2,torch
def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);inv=read(BASE/'inventory.json');start=time.monotonic();model=torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main','dinov2_vits14',source='local',pretrained=True).eval().cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None];total=0
 for cl in inv['clips']:
  cid=cl['clip'];dest=OUT/'patches'/f'{cid}.npz';dest.parent.mkdir(exist_ok=True)
  if dest.exists():continue
  entries=[e for e in inv['entries'] if e['clip']==cid];cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));uu=[];ff=[];xx=[];nn=[]
  for e in entries:
   ok,im=cap.read();assert ok;crops=[];positions=[]
   for t in e['targets']:
    x1,y1,x2,y2=map(int,t['box']);mask=decode(t['rle'])[y1:y2,x1:x2];crop=im[y1:y2,x1:x2].copy();crop[mask==0]=127;h,w=crop.shape[:2];s=224/max(h,w);ww,hh=max(1,round(w*s)),max(1,round(h*s));x,y=(224-ww)//2,(224-hh)//2;canvas=np.full((224,224,3),127,np.uint8);canvas[y:y+hh,x:x+ww]=cv2.resize(crop,(ww,hh));mc=np.zeros((224,224),np.uint8);mc[y:y+hh,x:x+ww]=cv2.resize(mask,(ww,hh),interpolation=cv2.INTER_NEAREST);occ=mc.reshape(16,14,16,14).mean((1,3));idx=np.flatnonzero(occ.reshape(-1)>=.6)
    if len(idx)>64:idx=idx[np.linspace(0,len(idx)-1,64).round().astype(int)]
    positions.append(idx);crops.append(canvas[:,:,::-1].copy());uu.append(t['target_key'])
   if crops:
    ten=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).cuda().float()/255
    with torch.inference_mode():z=model.forward_features((ten-mean)/std)['x_norm_patchtokens'];z=torch.nn.functional.normalize(z,dim=-1).cpu().numpy()
    for a,idx in zip(z,positions):
     feat=np.zeros((64,384),np.float32);xy=np.zeros((64,2),np.float32);n=len(idx);feat[:n]=a[idx];xy[:n]=np.column_stack([(idx%16+.5)/16,(idx//16+.5)/16]);ff.append(feat);xx.append(xy);nn.append(n)
  cap.release();np.savez_compressed(dest,uids=np.array(uu),features=np.array(ff),xy=np.array(xx),counts=np.array(nn));total+=len(uu);print('PATCHES',cid,len(uu),'supported',sum(n>=6 for n in nn),flush=True)
 write(OUT/'feature_complete.json',dict(passed=True,total_candidates=sum(len(np.load(OUT/'patches'/f'{c["clip"]}.npz')['uids']) for c in inv['clips']),seconds=time.monotonic()-start,weights_sha=sha('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')))
if __name__=='__main__':main()
