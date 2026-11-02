"""Decode every delivered frame, validate expected geometry and duration."""
from pathlib import Path
import json,cv2
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_g1_factor_supplement_20261006/presentation'

def main():
 manifest=json.loads((OUT/'media.json').read_text(encoding='utf8'))
 entries=[(r['path'],r['frames']) for r in manifest['clips']]
 entries.append((manifest['compilation'],sum(r['frames'] for r in manifest['clips'])))
 results=[]
 for name,expected in entries:
  cap=cv2.VideoCapture(str(OUT/name));assert cap.isOpened(),name
  fps=cap.get(cv2.CAP_PROP_FPS);n=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(1060,1280,3),(name,n,im.shape)
   n+=1
  cap.release();assert n==expected,(name,n,expected);assert abs(fps-10)<1e-6
  results.append(dict(file=name,frames_decoded=n,fps=fps,width=1280,height=1060))
 (OUT/'verification.json').write_text(json.dumps(dict(passed=True,files=results),indent=2),encoding='utf8')
 print('Media verified:',len(results),'files;',sum(r['frames_decoded'] for r in results),'decoded frames')
if __name__=='__main__':main()
