"""Inspect previously quarantined targets using raw seed crops, before masks."""
import json,collections
import cv2,numpy as np
from selection import ROOT,OUT,read,write
def main():
 selected=read(OUT/'selection.json');items=[]
 for c in selected:
  path=ROOT/'outputs'/c['source']/'mask_visual_review.json'
  if not path.exists():continue
  flags=read(path).get('quarantine_ids',{}).get(c['clip'],{})
  for e in c['entries']:
   if e['id'] not in flags:continue
   cap=cv2.VideoCapture(str(ROOT/c['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,e['anchor_frame'])
   ok,im=cap.read();cap.release();assert ok
   b=np.array(e['anchor_box'],int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,255),3)
   crop=im[max(0,b[1]-80):min(1920,b[3]+80),max(0,b[0]-80):min(2560,b[2]+80)]
   tile=np.full((320,260,3),245,np.uint8)
   factor=min(250/crop.shape[1],275/crop.shape[0])
   small=cv2.resize(crop,(max(1,int(crop.shape[1]*factor)),max(1,int(crop.shape[0]*factor))))
   tile[40:40+small.shape[0],5:5+small.shape[1]]=small
   label=c['id']+' '+e['segment']
   cv2.putText(tile,label,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
   cv2.putText(tile,'source '+c['clip']+' frame '+str(e['anchor_frame']),(6,35),cv2.FONT_HERSHEY_SIMPLEX,.38,(60,60,60),1,cv2.LINE_AA)
   items.append((dict(clip=c['id'],segment=e['segment'],source=c['source'],source_clip=c['clip'],
     frame=e['anchor_frame'],old_flag=flags[e['id']]),tile))
 dest=OUT/'presentation/seed_review';dest.mkdir(parents=True,exist_ok=True)
 for start in range(0,len(items),12):
  canvas=np.full((960,1040,3),255,np.uint8)
  for j,(_,tile) in enumerate(items[start:start+12]):
   y=(j//4)*320;x=(j%4)*260;canvas[y:y+320,x:x+260]=tile
  cv2.imwrite(str(dest/f'page_{start//12+1:02d}.jpg'),canvas)
 write(dest/'flagged_seed_inventory.json',[x for x,_ in items])
 print('flagged seed crops',len(items),'pages',(len(items)+11)//12)
if __name__=='__main__':main()
