"""Decode all delivered videos and verify their recorded dimensions and length."""
from qa import *
def main():
 dest=OUT/'presentation';checks=[]
 for path in sorted(dest.glob('*.mp4')):
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=0;shape=None
  while True:
   ok,im=cap.read()
   if not ok:break
   n+=1;shape=im.shape;assert im.shape[:2]==(1060,1280)
  cap.release();assert fps==10
  if path.name=='SELECTED_G1_WINDOWS.mp4':expected=630
  elif path.name=='ALL_REVIEWED_CLIPS.mp4':expected=1470
  else:
   cid=path.name[:2];expected=len(read_json(OUT/cid/'input/alignment.json'))
  assert n==expected,(path,n,expected)
  checks.append(dict(file=path.name,frames=n,fps=fps,decoded=True))
 assert len(checks)==22,len(checks);write_json(dest/'media_verified.json',dict(passed=True,checks=checks));print('Verified',len(checks),'videos',flush=True)
if __name__=='__main__':main()
