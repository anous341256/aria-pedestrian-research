from inspect_failures import *
import cv2,subprocess
PRES=OUT/'presentation'
CASES=[
 ('A','01',4,'N01','appearance','Appearance gate rejects a geometrically close match'),
 ('B','03',22,'N05','mixed','Motion gate favors stale track; source mask is ambiguous'),
 ('C','04',26,'N03','mixed','Two current candidates compete for one existing ID'),
 ('D','07',23,'N07','mixed','A stale track wins against the recent correct track'),
 ('E','07',26,'N04','motion','Uncertain 3D state falls back; old track is rejected'),
 ('F','07',28,'N06','mixed','A newly born wrong track accepts a distant person'),
 ('G','09',27,'N03','mixed','Night: malformed candidate and stale-track competition'),
 ('H','03',37,'N06','lifecycle','Candidate gap exceeds the track retention time')]
def txt(im,s,x,y,size=.55,col=(240,240,240)):
 cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(10,10,10),3,cv2.LINE_AA);cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,col,1,cv2.LINE_AA)
def fit(im,w,h):
 s=min(w/im.shape[1],h/im.shape[0]);p=cv2.resize(im,(max(1,round(im.shape[1]*s)),max(1,round(im.shape[0]*s))));can=np.full((h,w,3),25,np.uint8);y=(h-p.shape[0])//2;x=(w-p.shape[1])//2;can[y:y+p.shape[0],x:x+p.shape[1]]=p;return can
def crop(im,box):
 b=np.array(box);pad=np.array([max(30,.15*(b[2]-b[0])),max(30,.10*(b[3]-b[1]))]);lo=np.maximum(0,(b[:2]-pad).astype(int));hi=np.minimum([2560,1920],(b[2:]+pad).astype(int));return im[lo[1]:hi[1],lo[0]:hi[0]].copy()
def main():
 PRES.mkdir(parents=True,exist_ok=True);inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');catalog=read(OUT/'failure_catalog.json');selections=[]
 for code,cid,f,sid,kind,title in CASES:
  case=next(r for r in catalog if r['clip']==cid and r['frame']==f and r['source_id']==sid and (r['event']['type']=='mixed_reference_track' if kind=='mixed' else r['event']['type']=='reference_ID_change'));entries={e['frame']:e for e in inv['entries'] if e['clip']==cid};traces=read(OUT/'traces'/f'{cid}.json');lookup={r['uid']:r for t in traces for r in t['rows']};prevsource=case['selected_track_previous_source'] if kind=='mixed' else case['previous_person_track_source'];old=prevsource['source_id'] if prevsource else sid;last=prevsource['frame'] if prevsource else f-1
  times=[max(0,last-1),last,max(0,f-1),f,min(max(entries),f+1)];cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));frames={};needed=set(times)|set(range(max(0,min(last,f-5)),min(max(entries),f+5)+1))
  for j in range(max(needed)+1):
   ok,im=cap.read();assert ok
   if j in needed:frames[j]=im
  cap.release();sheet=np.full((980,1920,3),18,np.uint8);txt(sheet,f'{code} | Clip {cid} / frame {f} | {title}',15,30,.78);txt(sheet,'Yellow: previous source person. Cyan: current target. T: tracker output. Ref IDs are review-only.',15,60,.63)
  for j,t in enumerate(times):
   e=entries[t];im=frames[t];context=cv2.resize(im,(384,288));focus_sid=old if j<2 else sid;target=None
   for candidate in e['targets']:
    uid=candidate['target_key'];s=ref[uid]['source_id']
    if s not in [old,sid]:continue
    col=(0,220,255) if s==old and old!=sid else (255,220,0);b=np.asarray(candidate['box'])/ (2560/384);x1,y1,x2,y2=b.astype(int);cv2.rectangle(context,(x1,y1),(x2,y2),col,2);txt(context,f'{s}:T{lookup[uid]["track_id"]}',max(1,x1),max(16,y1),.4,col)
    if s==focus_sid:target=candidate
   sheet[90:378,j*384:(j+1)*384]=context;txt(sheet,f'Frame {t} | inspect {focus_sid}',j*384+8,405,.58)
   if target:
    sheet[420:730,j*384:(j+1)*384]=fit(crop(im,target['box']),384,310)
    r=lookup[target['target_key']];txt(sheet,f'T{r["track_id"]} / {r["action"]}',j*384+8,757,.64)
   else:txt(sheet,'No SAM candidate for this person',j*384+6,550,.47)
  q=case['previous_person_pair'];s=case['selected_pair'];lines=[]
  def desc(label,r):
   if r is None:return label+': no eligible previous-track pair'
   return f'{label}: distance {r["distance_px"]:.1f}px | appearance {r["appearance"]:.3f} | mahal2 {r["mahalanobis2"]:.2f} | cost {r["cost_unblocked"]:.3f} | {r["reason"]}'
  lines+=[desc('Previous target track',q),desc('Selected track',s),'Thresholds: appearance >=0.45; mahal2 <=13.82 unless appearance >=0.82; unmatched cost 0.72.','These are fixed-candidate diagnostics, not independent dense human-GT accuracy.']
  for j,line in enumerate(lines):txt(sheet,line,15,810+j*40,.65)
  cv2.imwrite(str(PRES/f'{code}_{cid}_{f:02d}_contact.jpg'),sheet)
  selections.append(dict(code=code,clip=cid,frame=f,source_id=sid,previous_source=old,previous_source_last_frame=last,type=kind,title=title,times=times,start_frame=min(needed),end_frame=max(needed),contact=f'{code}_{cid}_{f:02d}_contact.jpg',catalog_event=case))
 write(OUT/'selected_cases.json',selections)
 print([(v['code'],v['clip'],v['frame']) for v in selections])
if __name__=='__main__':main()
