from bulk_common import *
from render_bulk import cv2,color,text

inv=read(OUT/'inventory.json')
for cid,f,tid in [('022',20,6),('120',10,7),('104',0,1),('141',20,12)]:
 cl=next(c for c in inv['clips'] if c['clip']==cid);e=next(e for e in inv['entries'] if e['clip']==cid and e['frame']==f)
 row=next(r for r in read(OUT/'tracking'/f'{cid}.json')['rows'] if r['frame']==f and r['track_id']==tid);t=next(t for t in e['targets'] if t['target_key']==row['uid']);cap=cv2.VideoCapture(str(ROOT/cl['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,cl['lo']+f);ok,im=cap.read();cap.release();assert ok
 b=np.array(t['box']);pad=max(120,.5*max(b[2:]-b[:2]));a=np.maximum(b[:2]-pad,0).astype(int);z=np.minimum(b[2:]+pad,[2560,1920]).astype(int);crop=im[a[1]:z[1],a[0]:z[0]].copy();mask=decode(t['rle'])[a[1]:z[1],a[0]:z[0]]>0;crop[mask]=(.7*crop[mask]+.3*np.array([0,255,255])).astype(np.uint8);cv2.rectangle(crop,tuple((b[:2]-a).astype(int)),tuple((b[2:]-a).astype(int)),(0,255,255),2);text(crop,f'{cid} f{f} T{tid}',5,25,.7);dest=OUT/'review_flag_crops';dest.mkdir(exist_ok=True);cv2.imwrite(str(dest/f'{cid}_f{f}_T{tid}.jpg'),crop)
