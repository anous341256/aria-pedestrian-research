"""Keep raw discovery; remove redundant tile parts before provisional census."""
from detect import *
import argparse

def containment(a,b):
 a,b=np.array(a),np.array(b)
 inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
 return float(inter/max(1,np.prod(a[2:]-a[:2])))

def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+');args=p.parse_args()
 for c in read(OUT/'selected_clips.json'):
  cid=c['clip'];folder=OUT/cid;dest=folder/'refined';dest.mkdir(parents=True,exist_ok=True)
  if args.clips and cid not in args.clips:continue
  if (dest/'complete.json').exists() or not(folder/'discovery_complete.json').exists():continue
  ds=read(folder/'detections.json');filtered=[];pending=[]
  for fr in ds:
   detections=[r for r in fr['detections'] if r['confidence']>=.18];kept=[]
   for r in detections:
    larger=[q for q in detections if q is not r and q['confidence']>=.3 and
       np.prod(np.array(q['box'][2:])-q['box'][:2])>1.25*np.prod(np.array(r['box'][2:])-r['box'][:2]) and containment(r['box'],q['box'])>=.85]
    duplicate=[q for q in detections if q is not r and q['source']=='full' and iou(r['box'],q['box'])>=.40]
    if r['source']!='full' and (larger or duplicate):
     pending.append(dict(frame=fr['frame'],**r,reason='Redundant tile/partial-box proposal quarantined, not proof of absence',related=[q['detection_id'] for q in larger+duplicate]));continue
    kept.append(r)
   filtered.append(dict(frame=fr['frame'],detections=kept))
  cal=np.load(folder/'input/reference.npz');R,K=cal['R'],cal['K'];ki=np.linalg.inv(K);tracks=[]
  for fr in filtered:
   f=fr['frame'];det=fr['detections'];active=[t for t in tracks if f-t['rows'][-1]['frame']<=4];cost=np.ones((len(active),len(det)))
   for i,t in enumerate(active):
    last=t['rows'][-1];b=warp(last['box'],K@R[f].T@R[last['frame']]@ki)
    for j,d in enumerate(det):cost[i,j]=1-iou(b,d['box'])
   used=set()
   if cost.size:
    for i,j in zip(*linear_sum_assignment(cost)):
     if cost[i,j]<.82:active[i]['rows'].append(dict(frame=f,**det[j]));used.add(j)
   for j,d in enumerate(det):
    if j not in used:tracks.append(dict(proposal_id=len(tracks)+1,rows=[dict(frame=f,**d)]))
  near=[]
  for t in tracks:
   nr=[r for r in t['rows'] if r['box'][3]-r['box'][1]>=.12*1920 and r['confidence']>=.30]
   if not nr:continue
   rows=[r for r in t['rows'] if r['box'][3]-r['box'][1]>=.08*1920]
   near.append(dict(proposal_id=t['proposal_id'],rows=rows,first=rows[0]['frame'],last=rows[-1]['frame'],anchor=max(nr,key=lambda r:r['confidence']*math.sqrt(r['box'][3]-r['box'][1])),near_frames=[r['frame'] for r in nr]))
  write(dest/'near_proposals.json',near);write(dest/'all_tracklets.json',tracks);write(dest/'quarantined_detections.json',pending)
  # Reuse the discovery renderer but direct its outputs to refined/.
  cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'));frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(im)
  cap.release();sheets(cid+'/refined',frames,near)
  write(dest/'complete.json',dict(near=len(near),person_rows=sum(len(t['rows']) for t in near),quarantined=len(pending),method='Raw full-frame proposals retained; redundant tile detections quarantined; same rotation-assisted IoU association. Provisional IDs, no person cap.'))
  print(cid,len(read(folder/'near_proposals.json')),'->',len(near),'near',flush=True)
if __name__=='__main__':main()
