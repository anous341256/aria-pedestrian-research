from bulk_common import *
from render_bulk import cv2,text
inv=read(OUT/'inventory.json');lookup={t['target_key']:(e,t) for e in inv['entries'] for t in e['targets']};refs=read(OUT/'source_identity_reference.json');clips={c['clip']:c for c in inv['clips']};out=OUT/'mixed_window_review';out.mkdir(exist_ok=True);index=[]
for w in read(OUT/'forecast_window_candidates.json')['windows']:
 if w['source_identity_consistent_review_only']:continue
 ids=[refs[u]['source_id'] for u in w['uids']];changes=[j for j in range(1,21) if ids[j]!=ids[j-1]];sel=sorted(set([0,10,20]+[k for j in changes for k in [max(0,j-1),j,min(20,j+1)]]));canvas=np.zeros((((len(sel)+3)//4)*310,1200,3),np.uint8);cl=clips[w['clip']];cap=cv2.VideoCapture(str(ROOT/cl['input']/'rgb.mp4'))
 for pos,j in enumerate(sel):
  e,t=lookup[w['uids'][j]];cap.set(cv2.CAP_PROP_POS_FRAMES,e['source_frame']);ok,im=cap.read();assert ok;b=np.array(t['box']);pad=max(60,.25*max(b[2:]-b[:2]));a=np.maximum(b[:2]-pad,0).astype(int);z=np.minimum(b[2:]+pad,[2560,1920]).astype(int);crop=im[a[1]:z[1],a[0]:z[0]].copy();mask=decode(t['rle'])[a[1]:z[1],a[0]:z[0]]>0;crop[mask]=(.80*crop[mask]+.20*np.array([0,255,255])).astype(np.uint8);s=min(300/crop.shape[1],280/crop.shape[0]);crop=cv2.resize(crop,(max(1,round(crop.shape[1]*s)),max(1,round(crop.shape[0]*s))));y=(pos//4)*310;x=(pos%4)*300;canvas[y+30:y+30+crop.shape[0],x:x+crop.shape[1]]=crop;text(canvas,f'{w["key"]} f{j} src{ids[j]}',x+3,y+20,.43)
 cap.release();cv2.imwrite(str(out/f'{w["key"]}.jpg'),canvas);index.append(dict(key=w['key'],quality_eligible=w['quality_eligible'],source_ids=ids,change_frames=changes,review_frames=sel))
write(out/'index.json',index);print('MIXED',len(index))
