"""Inspection sheets and descriptive disagreement, never accuracy against SAM."""
import argparse,collections,json
from pathlib import Path
import cv2,numpy as np
from render import ROOT,OUT,read,load,decode
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 cs=read(OUT/'selection.json');cs=[c for c in cs if (OUT/c['id']/'complete.json').exists()]
 if a.limit:cs=cs[:a.limit]
 dest=OUT/'presentation/inspection';dest.mkdir(parents=True,exist_ok=True)
 rows=[]
 for c in cs:
  series={}
  for arm in ['visual','camera_person']:
   ll=load(OUT/c['id']/arm/'decisions.jsonl.gz')
   series[arm]={(x['segment'],x['frame']):x for x in ll if 'raw_rle' in x}
  values=[]
  for key in series['visual'].keys()&series['camera_person'].keys():
   x,y=series['visual'][key],series['camera_person'][key]
   # Tight small masks suffice for visual disagreement diagnostics only.
   m=cv2.resize(decode(x['raw_rle']).astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST)>0
   n=cv2.resize(decode(y['raw_rle']).astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST)>0
   u=np.count_nonzero(m|n)
   disagreement=1-np.count_nonzero(m&n)/u if u else 0.
   values.append((disagreement,key[1],key[0],x['status'],y['status']))
  if values:
   top=max(values)
   rows.append(dict(clip=c['id'],recording=c['recording'],frame=top[1],segment=top[2],
        disagreement=top[0],visual_status=top[3],internal_status=top[4]))
 rows.sort(key=lambda x:x['disagreement'],reverse=True)
 (dest/'disagreement_cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 lookup={c['id']:c for c in cs}
 for r in rows[:12]:
  c=lookup[r['clip']];path=OUT/'presentation/raw'/(c['id']+'.mp4')
  cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']-c['lo'])
  ok,im=cap.read();cap.release()
  if ok:cv2.imwrite(str(dest/(r['clip']+'_largest_disagreement.jpg')),im)
 print(json.dumps(rows[:12],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
