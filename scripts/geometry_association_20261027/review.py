from assoc_common import *
import cv2,collections
def text(im,s,x,y,size=.48,col=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,col,1,cv2.LINE_AA)
def main():
 inv=read(OUT/'inventory.json');ref=read(OUT/'source_identity_reference.json');comp=read(OUT/'comparison.json');stats={}
 for cond in ['natural','blackout3']:
  stats[cond]={}
  for a in ARMS:
   mix=[]
   for cl in inv['clips']:
    rr=read(OUT/'tracking'/cond/a/f'{cl["clip"]}.json')['rows'];groups=collections.defaultdict(list)
    for r in rr:groups[r['track_id']].append(r)
    for tid,v in groups.items():
     for x,y in zip(v[:-1],v[1:]):
      if ref[x['uid']]['source_id']!=ref[y['uid']]['source_id']:mix.append(dict(clip=cl['clip'],track_id=tid,frame=y['frame'],previous_frame=x['frame'],previous_source=ref[x['uid']]['source_id'],current_source=ref[y['uid']]['source_id']))
   stats[cond][a]=dict(within_track_reference_identity_changes=len(mix),events=mix)
 write(OUT/'identity_diagnostics.json',stats)
 # Targeted contrasts plus a one-person failure. Selection is diagnostic, not an evaluation subset.
 cases=[('07','N05',23,'Full geometry avoids a reference ID change'),('07','N04',26,'Negative contrast: full geometry changes ID'),('09','N02',38,'Full geometry avoids a reference ID change'),('10','N02',8,'Full geometry avoids a reference ID change'),('01','N01',None,'Single person: fragmentation is not identity crossing')]
 notes=[];PRES=OUT/'presentation';PRES.mkdir(exist_ok=True)
 for num,(cid,sid,frame,label) in enumerate(cases):
  if frame is None:frame=comp['natural']['rotation']['per_clip'][cid]['changes'][0]['frame']
  entries={e['frame']:e for e in inv['entries'] if e['clip']==cid};rr={a:{r['uid']:r for r in read(OUT/'tracking'/'natural'/a/f'{cid}.json')['rows']} for a in ARMS};times=[max(0,frame-3),max(0,frame-1),frame,min(max(entries),frame+1),min(max(entries),frame+3)]
  canvas=np.full((440,1500,3),22,np.uint8);text(canvas,f'{cid} / {sid}: {label}',15,25,.68);cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));details=[]
  for j,f in enumerate(times):
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;target=next((t for t in entries[f]['targets'] if ref[t['target_key']]['source_id']==sid),None)
   if target is None:text(canvas,f'Frame {f}: no candidate',j*300+10,70);continue
   b=np.asarray(target['box'],int);w=b[2]-b[0];h=b[3]-b[1];x1,y1=np.maximum([0,0],b[:2]-[int(.3*w),int(.15*h)]);x2,y2=np.minimum([2560,1920],b[2:]+[int(.3*w),int(.15*h)]);crop=im[y1:y2,x1:x2].copy();cv2.rectangle(crop,(b[0]-x1,b[1]-y1),(b[2]-x1,b[3]-y1),(0,210,255),2);sc=min(280/crop.shape[1],270/crop.shape[0]);crop=cv2.resize(crop,(int(crop.shape[1]*sc),int(crop.shape[0]*sc)));canvas[50:50+crop.shape[0],j*300+10:j*300+10+crop.shape[1]]=crop;text(canvas,f'Frame {f}',j*300+10,340,.6)
   info={}
   for k,a in enumerate(ARMS):
    r=rr[a][target['target_key']];text(canvas,f'{["Image","Rotation","Full3D"][k]}: T{r["track_id"]} {r["action"]}',j*300+10,365+k*23,.48);info[a]=r
   details.append(dict(frame=f,uid=target['target_key'],associations=info))
  cap.release();file=f'contrast_{num+1:02d}_{cid}_{sid}.jpg';cv2.imwrite(str(PRES/file),canvas);notes.append(dict(clip=cid,source_id=sid,frame=frame,selection=label,image=file,details=details))
 write(OUT/'review_cases.json',notes)
 # Fixed midframes from every clip, irrespective of outcome.
 sheet=np.full((5*320,2*960,3),15,np.uint8)
 for k,cl in enumerate(inv['clips']):
  path=PRES/f'{cl["clip"]}_midframe.jpg'
  if path.exists():sheet[(k//2)*320:(k//2+1)*320,(k%2)*960:(k%2+1)*960]=cv2.resize(cv2.imread(str(path)),(960,320))
 cv2.imwrite(str(PRES/'all10_midframes.jpg'),sheet)
 print({c:{a:v['within_track_reference_identity_changes'] for a,v in vals.items()} for c,vals in stats.items()})
if __name__=='__main__':main()
