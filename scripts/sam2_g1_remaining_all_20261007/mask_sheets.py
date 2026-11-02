"""Per-identity sampled mask and RGB crops, including weakest retained prompt agreement."""
from audit import *
import argparse
cv2.setNumThreads(2)

def process(cid):
 folder=OUT/cid;dest=OUT/'mask_review';dest.mkdir(exist_ok=True)
 done=dest/f'{cid}_index.json'
 if done.exists():return
 rows=v.load(folder/'review_v1/masks_rle.jsonl.gz');by=collections.defaultdict(list)
 for r in rows:by[r['id']].append(r)
 if not by:write(done,[]);return
 samples={};frames={};cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'))
 for pid,rr in sorted(by.items()):
  rr.sort(key=lambda r:r['frame']);ss=[rr[int(round(t*(len(rr)-1)))] for t in np.linspace(0,1,4)]
  ss.append(min(rr,key=lambda r:r['quality']['guide_iou']));samples[pid]=ss
 wanted={r['frame'] for ss in samples.values() for r in ss};f=0
 while cap.isOpened():
  ok,im=cap.read()
  if not ok:break
  if f in wanted:frames[f]=im
  f+=1
 cap.release();strips=[]
 for pid,ss in samples.items():
  strip=np.full((225,1500,3),245,np.uint8)
  for j,r in enumerate(ss):
   im=frames[r['frame']].copy();m=v.decode(r['rle']);b=np.array(r['box'],float);pad=max(30,.15*(b[3]-b[1]));x1,y1,x2,y2=np.rint(b+[-pad,-pad,pad,pad]).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(2560,x2);y2=min(1920,y2)
   color=np.array(v.color(int(pid[1:])-1));mm=m.astype(bool);im[mm]=(.65*im[mm]+.35*color).astype(np.uint8)
   cc,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cc,-1,tuple(map(int,color)),2)
   crop=im[y1:y2,x1:x2];h,w=crop.shape[:2];s=min(296/w,190/h);crop=cv2.resize(crop,(max(1,int(w*s)),max(1,int(h*s))))
   strip[28:28+crop.shape[0],j*300:j*300+crop.shape[1]]=crop
   cv2.putText(strip,f'{cid}/{pid} f{r["frame"]}'+(' weak' if j==4 else ''),(j*300+3,19),0,.5,(20,20,20),1,cv2.LINE_AA)
  strips.append((pid,strip))
 index=[]
 for k in range(0,len(strips),6):
  ss=strips[k:k+6];p=dest/f'{cid}_{k//6+1:02d}.jpg';cv2.imwrite(str(p),np.vstack([x[1] for x in ss]),[cv2.IMWRITE_JPEG_QUALITY,94]);index.append(dict(page=p.name,ids=[x[0] for x in ss]))
 write(done,index);print(cid,'MASK_SHEETS',len(index),len(by),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
 for c in read(OUT/'selected_clips.json'):
  if a.wait:
   while not(OUT/c['clip']/'review_v1/audit.json').exists():time.sleep(2)
  if (OUT/c['clip']/'review_v1/audit.json').exists():process(c['clip'])
