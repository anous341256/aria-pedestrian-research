"""All-person short survey proposals; no learned forecast evaluation."""
from pathlib import Path
import json,time,hashlib
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def iou(a,b):
 a,b=np.array(a),np.array(b);inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
 return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def warp(b,H):
 p=H@np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]]).T;p=p[:2]/p[2:]
 return np.r_[p.min(1),p.max(1)]
def nms(rows):
 out=[]
 for r in sorted(rows,key=lambda r:-r['confidence']):
  if all(iou(r['box'],q['box'])<.6 for q in out):out.append(r)
 return out
def main():
 model=YOLO(str(ROOT/'models/tracking/yolo11l.pt'));start=time.monotonic();summaries=[]
 for sample in read(OUT/'sample_locked.json'):
  dest=OUT/'samples'/sample['sample_id']
  if (dest/'proposals.json').exists():continue
  while not (dest/'ready.json').exists():time.sleep(1)
  ims=[cv2.imread(str(dest/f'f{f:02d}.jpg')) for f in range(21)];assert all(im is not None for im in ims)
  ds=[]
  for f in range(21):
   windows=[(0,0,2560,1920)]
   if f==5:windows += [(0,0,1600,1200),(960,0,2560,1200),(0,720,1600,1920),(960,720,2560,1920)]
   rows=[]
   for j,(x1,y1,x2,y2) in enumerate(windows):
    res=model.predict(ims[f][y1:y2,x1:x2],classes=[0],imgsz=1536 if j==0 else 1280,conf=.12,iou=.65,max_det=1000,device=0,half=True,verbose=False)[0]
    for b,p in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy()):
     rows.append(dict(box=(b+np.array([x1,y1,x1,y1])).tolist(),confidence=float(p),source='full' if j==0 else f'tile{j}'))
   ds.append(nms(rows))
  write(dest/'detections.json',ds);cal=np.load(dest/'reference.npz');R,K=cal['R'],cal['K'];ki=np.linalg.inv(K);tracks=[]
  for f,det0 in enumerate(ds):
   det=[d for d in det0 if d['confidence']>=.18];active=[t for t in tracks if f-t['rows'][-1]['frame']<=2]
   cost=np.ones((len(active),len(det)))
   for i,t in enumerate(active):
    r=t['rows'][-1];b=warp(r['box'],K@R[f].T@R[r['frame']]@ki)
    for j,d in enumerate(det):cost[i,j]=1-iou(b,d['box'])
   assigned=set()
   if cost.size:
    for i,j in zip(*linear_sum_assignment(cost)):
     if cost[i,j]<.82:active[i]['rows'].append(dict(frame=f,**det[j]));assigned.add(j)
   for j,d in enumerate(det):
    if j not in assigned:tracks.append(dict(track=len(tracks)+1,rows=[dict(frame=f,**d)]))
  candidates=[]
  for t in tracks:
   at=next((r for r in t['rows'] if r['frame']==5),None)
   if at is None:continue
   b=np.array([r['box'] for r in t['rows']]);height=(b[:,3]-b[:,1])/1920
   reasons=[]
   if len(t['rows'])!=21:reasons.append('not_continuous21')
   if height.min()<.04:reasons.append('below4percent')
   if np.any(b[:,0]<=5)|np.any(b[:,1]<=5)|np.any(b[:,2]>=2555)|np.any(b[:,3]>=1915):reasons.append('border')
   if max(r['confidence'] for r in t['rows'])<.35:reasons.append('weak_detector')
   # Intersection relative to smaller detected person; leave final contextual decision to review.
   overlap=[]
   for r in t['rows']:
    own=np.array(r['box']);area=np.prod(own[2:]-own[:2]);mx=0.
    for other in ds[r['frame']]:
     ob=np.array(other['box'])
     if iou(ob,own)>.9:continue
     inter=np.prod(np.maximum(0,np.minimum(own[2:],ob[2:])-np.maximum(own[:2],ob[:2])))
     mx=max(mx,float(inter/max(1,area)))
    overlap.append(mx)
   candidates.append(dict(**t,key=f'{sample["sample_id"]}_T{t["track"]}',sample_id=sample['sample_id'],
                         automatic_reasons=reasons,automatic_eligible=not reasons,min_height=float(height.min()),
                         max_overlap=float(max(overlap)),overlap_frames=sum(x>.25 for x in overlap),
                         near_candidate=bool(height.min()>=.08),cutoff=at))
  write(dest/'proposals.json',candidates)
  # Full-frame review always retains all sampled windows, including empty/failed ones.
  sheet=Image.new('RGB',(1600,280),(245,245,245));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
  for j,f in enumerate([0,5,10,15,20]):
   im=ims[f].copy()
   for t in candidates:
    r=next((r for r in t['rows'] if r['frame']==f),None)
    if r is None:continue
    b=np.round(r['box']).astype(int);col=(40,220,40) if t['automatic_eligible'] else (60,170,240)
    cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),col,4);cv2.putText(im,str(t['track']),tuple(b[:2]+[0,-7]),0,1.3,(255,50,50),4)
   sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(320,240)),cv2.COLOR_BGR2RGB)),(j*320,40))
  draw.text((4,2),f'{sample["sample_id"]} {sample["id"]} {sample["start_s"]:.1f}s | cutoff people {len(candidates)} | full {sum(t["automatic_eligible"] for t in candidates)}',font=font,fill='black')
  sheet.save(dest/'context.jpg',quality=94)
  for t in candidates:
   if not t['automatic_eligible']:continue
   sheet=Image.new('RGB',(1470,230),(245,245,245));draw=ImageDraw.Draw(sheet)
   for j,f in enumerate([0,2,5,8,12,16,20]):
    r=next(r for r in t['rows'] if r['frame']==f);x1,y1,x2,y2=np.round(r['box']).astype(int);pad=max(15,int((y2-y1)*.15))
    crop=Image.fromarray(cv2.cvtColor(ims[f][max(0,y1-pad):min(1920,y2+pad),max(0,x1-pad):min(2560,x2+pad)],cv2.COLOR_BGR2RGB));crop.thumbnail((205,188));sheet.paste(crop,(j*210+(205-crop.width)//2,40))
   draw.text((4,2),f'{t["key"]} | minH {t["min_height"]:.3f} | overlap {t["max_overlap"]:.2f}',font=font,fill='black')
   sheet.save(dest/f'track_{t["track"]:03d}.jpg',quality=94)
  print(sample['sample_id'],'cutoff',len(candidates),'eligible',sum(t['automatic_eligible'] for t in candidates),'elapsed',round(time.monotonic()-start,1),flush=True)
 write(OUT/'detector_complete.json',dict(samples=160,seconds=time.monotonic()-start,weights_sha256=hashlib.sha256((ROOT/'models/tracking/yolo11l.pt').read_bytes()).hexdigest()))
if __name__=='__main__':main()
