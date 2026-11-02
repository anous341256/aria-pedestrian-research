from bulk_common import *
from render_bulk import cv2,color,text
inv=read(OUT/'inventory.json')
for cid,tid in [('022',6),('120',7),('104',1)]:
 cl=next(c for c in inv['clips'] if c['clip']==cid);rr=[r for r in read(OUT/'tracking'/f'{cid}.json')['rows'] if r['track_id']==tid];chosen=[rr[0],rr[len(rr)//2],rr[-1]];canvas=np.zeros((750,1500,3),np.uint8)
 cap=cv2.VideoCapture(str(ROOT/cl['input']/'rgb.mp4'))
 for j,r in enumerate(chosen):
  f=r['frame'];e=next(e for e in inv['entries'] if e['clip']==cid and e['frame']==f);t=next(t for t in e['targets'] if t['target_key']==r['uid']);cap.set(cv2.CAP_PROP_POS_FRAMES,cl['lo']+f);ok,im=cap.read();assert ok;b=np.array(t['box']);pad=max(100,.5*max(b[2:]-b[:2]));a=np.maximum(b[:2]-pad,0).astype(int);z=np.minimum(b[2:]+pad,[2560,1920]).astype(int);crop=im[a[1]:z[1],a[0]:z[0]].copy();overlay=crop.copy();mask=decode(t['rle'])[a[1]:z[1],a[0]:z[0]]>0;overlay[mask]=(.7*overlay[mask]+.3*np.array([0,255,255])).astype(np.uint8)
  for k,img in enumerate([crop,overlay]):
   s=min(500/img.shape[1],345/img.shape[0]);img=cv2.resize(img,(max(1,round(s*img.shape[1])),max(1,round(s*img.shape[0]))));canvas[30+k*360:30+k*360+img.shape[0],j*500:j*500+img.shape[1]]=img
  text(canvas,f'{cid} T{tid} frame {f}',j*500+5,20,.6)
 cap.release();cv2.imwrite(str(OUT/'review_flag_crops'/f'{cid}_T{tid}_temporal.jpg'),canvas)
