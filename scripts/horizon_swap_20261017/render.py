"""Same nine previously fixed cases,seed17; side-by-side matched future frames."""
from prepare import *
import subprocess
PRES=OUT/'presentation'
def main():
 PRES.mkdir(parents=True,exist_ok=True);d=np.load(OUT/'dataset.npz');meta=read(OUT/'samples.json');keys=read(OLD/'presentation/selection.json')['fixed'];lookup={m['key']:i for i,m in enumerate(meta)};pred={}
 for fold in read(OUT/'folds.json'):
  for arm in ['short_history_control','long_history']:
   z=np.load(OUT/arm/f'fold_{fold["fold"]}/seed_17/person/predictions.npz')
   for j,i in enumerate(z['indices']):pred.setdefault(int(i),{})[arm]=(z['image'][j],z['image_error'][j].mean())
 dest=PRES/'fixed_9_short_vs_long.mp4';ffmpeg='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';destwin='E:/'+str(dest).removeprefix('/mnt/e/');args=[ffmpeg,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x620','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',destwin]
 if dest.exists():raise RuntimeError('Preserve previous render')
 log=(PRES/'render.log').open('wb');proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
 def txt(im,s,x,y,size=.7,color=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)
 def pt(im,p,color,r=7):
  q=np.asarray(p)*.25
  if np.isfinite(q).all():cv2.circle(im,tuple(np.clip(q,[-1000,-1000],[1640,1480]).astype(int)),r,color,2,cv2.LINE_AA)
 count=0
 for key in keys:
  i=lookup[key];m=meta[i];video=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4';cap=cv2.VideoCapture(str(video));assert cap.isOpened();frames=[]
  cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][0])
  for j in range(21):
   ok,im=cap.read();assert ok;frames.append(cv2.resize(im,(640,480)))
  cap.release();truth=d['xy'][i,16:21];hold=np.repeat(d['xy'][i,15:16],5,0);ha=np.linalg.norm(hold-truth,axis=-1).mean()
  for j,im in enumerate(frames):
   out=np.zeros((620,1920,3),np.uint8);txt(out,f'{key} | Same forecast origin and future frames | Seed17 | Half speed',14,30,.8)
   for col,(name,arr,ade,beg) in enumerate([('0.5s history / 0.5s future',*pred[i]['short_history_control'],10),('1.5s history / 0.5s future',*pred[i]['long_history'],0),('Hold last position',hold,ha,15)]):
    panel=im.copy();pt(panel,d['xy'][i,j],(255,255,255),6)
    if j<=15 and j>=beg:
     for t in range(beg,j+1):pt(panel,d['xy'][i,t],(255,200,60),3)
    if j>15:
     pt(panel,arr[j-16],(0,160,255),9);a=np.clip(truth[j-16]*.25,-5000,5000).astype(int);b=np.clip(arr[j-16]*.25,-5000,5000).astype(int);cv2.line(panel,tuple(a),tuple(b),(0,160,255),2,cv2.LINE_AA)
    out[88:568,col*640:(col+1)*640]=panel;txt(out,name+f' | ADE {ade:.1f}px',col*640+10,65,.68)
   phase=f'History: t={(j-15)*.1:+.1f}s' if j<=15 else f'Fixed forecast: +{(j-15)*.1:.1f}s (no new observations)'
   txt(out,phase+' | White: SAM reference center. Orange: forecast.',14,605,.76)
   proc.stdin.write(out.tobytes());proc.stdin.write(out.tobytes());count+=2
   if j==20:cv2.imwrite(str(PRES/f'{key}.jpg'),out)
 proc.stdin.close();assert proc.wait(timeout=60)==0;log.close();cap=cv2.VideoCapture(str(dest));decoded=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape[:2]==(620,1920);decoded+=1
 cap.release();assert decoded==count==378
 write(PRES/'verification.json',dict(passed=True,frames=decoded,seconds=37.8,fixed_keys=keys,selection='Same original nine cases selected before this experiment; seed17; no score-based replacement.'))
 print(str(dest),flush=True)
if __name__=='__main__':main()
