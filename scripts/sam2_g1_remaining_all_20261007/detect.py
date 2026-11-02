"""All-frame near-person discovery and provisional association for review."""
from pathlib import Path
import sys,json,time,math
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_remaining_all_20261007'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json as read,write_json as write
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
def iou(a,b):
 a,b=np.array(a),np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def warp(b,H):
 a=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]]).T;q=H@a;q=q[:2]/q[2:];return np.r_[q.min(1),q.max(1)]
def nms(rows):
 out=[]
 for r in sorted(rows,key=lambda r:-r['confidence']):
  if all(iou(r['box'],q['box'])<.6 for q in out):out.append(r)
 return out
def sheets(cid,frames,candidates):
 d=OUT/cid;context=Image.new('RGB',(1600,280),(245,245,245));draw=ImageDraw.Draw(context)
 for j,f in enumerate(np.linspace(0,len(frames)-1,5,dtype=int)):
  im=frames[f].copy()
  for t in candidates:
   r=next((r for r in t['rows'] if r['frame']==f),None)
   if r:
    b=np.array(r['box'],int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,220,220),3);cv2.putText(im,f"P{t['proposal_id']}",tuple(b[:2]+[0,-8]),0,1.3,(50,50,255),3)
  context.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(320,240)),cv2.COLOR_BGR2RGB)),(j*320,40))
 draw.text((4,4),f'{cid} | frames {len(frames)} | near proposals {len(candidates)}',font=FONT,fill='black');context.save(d/'context.jpg',quality=94)
 for t in candidates:
  canvas=Image.new('RGB',(1400,210),(245,245,245));draw=ImageDraw.Draw(canvas)
  for j,k in enumerate(np.unique(np.linspace(0,len(t['rows'])-1,6,dtype=int))):
   r=t['rows'][k];b=np.array(r['box'],int);pad=max(12,int(.12*(b[3]-b[1])));x1,y1=np.maximum(b[:2]-pad,0);x2,y2=np.minimum(b[2:]+pad,[2560,1920])
   im=Image.fromarray(cv2.cvtColor(frames[r['frame']][y1:y2,x1:x2],cv2.COLOR_BGR2RGB));im.thumbnail((200,175));canvas.paste(im,(180+j*202,0));draw.text((180+j*202,182),f'f{r["frame"]}',font=FONT,fill='black')
  draw.text((4,3),f'{cid} P{t["proposal_id"]}\nf{t["first"]}-{t["last"]}\nn={len(t["rows"])}',font=FONT,fill='black');canvas.save(d/f'track_{t["proposal_id"]:03d}.jpg',quality=93)
def main():
 model=YOLO(str(ROOT/'models/tracking/yolo11l.pt'));begin=time.monotonic();totals=[]
 for c in read(OUT/'selected_clips.json'):
  cid=c['clip'];dest=OUT/cid
  if (dest/'discovery_complete.json').exists():totals.append(read(dest/'discovery_complete.json'));continue
  while not (dest/'input/metadata.json').exists():time.sleep(1)
  cap=cv2.VideoCapture(str(dest/'input/rgb.mp4'));frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(im)
  cap.release();assert len(frames)==read(dest/'input/metadata.json')['frames']
  if (dest/'detections.json').exists():ds=read(dest/'detections.json')
  else:
   ds=[]
   for f,im in enumerate(frames):
    crops=[(0,0,2560,1920)]
    if f%5==0:crops += [(0,0,1600,1200),(960,0,2560,1200),(0,720,1600,1920),(960,720,2560,1920)]
    rows=[]
    for j,(x1,y1,x2,y2) in enumerate(crops):
     res=model.predict(im[y1:y2,x1:x2],classes=[0],imgsz=1536 if j==0 else 1280,conf=.12,iou=.65,max_det=1000,device=0,half=True,verbose=False)[0]
     for b,p in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy()):rows.append(dict(box=(b+[x1,y1,x1,y1]).tolist(),confidence=float(p),source='full' if j==0 else f'tile{j}'))
    rows=nms(rows)
    for j,r in enumerate(rows):r['detection_id']=f'{f:03d}_{j:03d}'
    ds.append(dict(frame=f,detections=rows))
   write(dest/'detections.json',ds)
  cal=np.load(dest/'input/reference.npz');R,K=cal['R'],cal['K'];ki=np.linalg.inv(K);tracks=[]
  for row in ds:
   f=row['frame'];det=[d for d in row['detections'] if d['confidence']>=.18];active=[t for t in tracks if f-t['rows'][-1]['frame']<=4];cost=np.ones((len(active),len(det)))
   for i,t in enumerate(active):
    last=t['rows'][-1];b=warp(last['box'],K@R[f].T@R[last['frame']]@ki)
    for j,d in enumerate(det):cost[i,j]=1-iou(b,d['box'])
   assigned=set()
   if cost.size:
    for i,j in zip(*linear_sum_assignment(cost)):
     if cost[i,j]<.82:active[i]['rows'].append(dict(frame=f,**det[j]));assigned.add(j)
   for j,d in enumerate(det):
    if j not in assigned:tracks.append(dict(proposal_id=len(tracks)+1,rows=[dict(frame=f,**d)]))
  candidates=[]
  for t in tracks:
   near=[r for r in t['rows'] if r['box'][3]-r['box'][1]>=.12*1920 and r['confidence']>=.30]
   if not near:continue
   rows=[r for r in t['rows'] if r['box'][3]-r['box'][1]>=.08*1920]
   candidates.append(dict(proposal_id=t['proposal_id'],rows=rows,first=rows[0]['frame'],last=rows[-1]['frame'],anchor=max(near,key=lambda r:r['confidence']*math.sqrt(r['box'][3]-r['box'][1])),near_frames=[r['frame'] for r in near]))
  write(dest/'proposal_tracklets.json',tracks);write(dest/'near_proposals.json',candidates);sheets(cid,frames,candidates)
  status=dict(clip=cid,frames=len(frames),proposals=len(tracks),near_proposals=len(candidates),seconds=time.monotonic()-begin);write(dest/'discovery_complete.json',status);totals.append(status)
  print(cid,'DISCOVERED',len(candidates),'near',len(frames),'frames','elapsed',round(time.monotonic()-begin,1),flush=True)
 write(OUT/'discovery_complete.json',totals)
if __name__=='__main__':main()
