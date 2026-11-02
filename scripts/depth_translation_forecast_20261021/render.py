from depth_common import *
import cv2,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PRES=OUT/'presentation'
def main():
 PRES.mkdir(parents=True,exist_ok=True);data=np.load(BASE/'dataset.npz');meta=read(BASE/'samples.json');keys=read(ROOT/'outputs/biflow_aria_v1_20261002/presentation/selection.json')['fixed'];lookup={m['key']:i for i,m in enumerate(meta)};pred={};depth=np.load(OUT/'depth_features.npz')['features']
 for fold in read(BASE/'folds.json'):
  for arm in ['saved_translation','no_depth','with_depth']:
   path=BASE/'with_translation'/f'fold_{fold["fold"]}/seed_17/person/predictions.npz' if arm=='saved_translation' else OUT/arm/f'fold_{fold["fold"]}/seed_17/predictions.npz'
   z=np.load(path)
   for j,i in enumerate(z['indices']):pred.setdefault(int(i),{})[arm]=(z['image'][j],float(z['image_error'][j].mean()))
 dest=PRES/'fixed_9_depth_comparison.mp4';ff='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';win='E:/'+str(dest).removeprefix('/mnt/e/')
 assert not dest.exists();log=(PRES/'render.log').open('wb');proc=subprocess.Popen([ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',win],stdin=subprocess.PIPE,stderr=log)
 def txt(im,s,x,y,size=.65,c=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,2,cv2.LINE_AA)
 def dot(im,p,c,r=5):
  q=np.asarray(p)*.25
  if np.isfinite(q).all():cv2.circle(im,tuple(np.clip(q,-5000,5000).astype(int)),r,c,2,cv2.LINE_AA)
 count=0
 for key in keys:
  i=lookup[key];m=meta[i];cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][0]);frames=[]
  for j in range(21):
   ok,im=cap.read();assert ok;frames.append(cv2.resize(im,(640,480)))
  cap.release();valid=depth[i,:,2]>0;zs=np.where(valid,30**depth[i,:,0],np.nan)
  for j,im in enumerate(frames):
   out=np.zeros((720,1920,3),np.uint8);txt(out,f'{key} | Fixed original cases | Seed17 | History 1.5s -> future 0.5s',14,30,.8)
   for col,(arm,label) in enumerate([('saved_translation','Saved translation model'),('no_depth','Matched fine-tune: no depth'),('with_depth','Matched fine-tune: with depth')]):
    p,ade=pred[i][arm];panel=im.copy();dot(panel,data['xy'][i,j],(255,255,255))
    if j<=15:
     for t in range(j+1):dot(panel,data['xy'][i,t],(255,200,60),3)
    else:
     dot(panel,p[j-16],(0,160,255),9);cv2.line(panel,tuple(np.clip(data['xy'][i,j]*.25,-5000,5000).astype(int)),tuple(np.clip(p[j-16]*.25,-5000,5000).astype(int)),(0,160,255),2,cv2.LINE_AA)
    out[88:568,col*640:(col+1)*640]=panel;txt(out,label+f' | ADE {ade:.1f}px',col*640+10,65,.60)
   phase=f'Observed history {(j-15)*.1:+.1f}s' if j<=15 else f'Fixed forecast +{(j-15)*.1:.1f}s; no new measurements'
   txt(out,phase+' | White: reference center; orange: prediction',14,600,.75)
   txt(out,f'Historical depth: {int(valid.sum())}/16 valid. Same camera predictions in all panels.',14,635,.72)
   for k in range(16):
    x=20+k*114;color=(95,205,100) if valid[k] else (110,110,110);cv2.rectangle(out,(x,650),(x+105,708),color,2)
    txt(out,f'{zs[k]:.2f}m' if valid[k] else 'missing',x+3,686,.46,color)
   proc.stdin.write(out.tobytes());proc.stdin.write(out.tobytes());count+=2
   if j==20:cv2.imwrite(str(PRES/f'{key}.jpg'),out)
 proc.stdin.close();assert proc.wait(timeout=60)==0;log.close();cap=cv2.VideoCapture(str(dest));decoded=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape[:2]==(720,1920);decoded+=1
 cap.release();assert decoded==count==378
 c=read(OUT/'comparison.json')['results'];names=['saved_translation','no_depth','with_depth','hold_last'];labels=['Saved translation','Matched no depth','With depth','Hold last'];fig,axs=plt.subplots(1,2,figsize=(11,4),dpi=160)
 for ax,metric in zip(axs,['ADE','FDE']):
  vals=[c[n]['window_mean'][metric] for n in names];ax.bar(labels,vals,color=['#577590','#999999','#43aa8b','#edb458']);ax.set_ylabel(metric+' (pixels, lower is better)');ax.tick_params(axis='x',rotation=15)
  for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
  ax.set_ylim(0,max(vals)*1.17)
 fig.suptitle('Historical depth as person-branch features | 117 windows, 6 recording folds, 3 seeds');fig.tight_layout();fig.savefig(PRES/'comparison.png');fig.savefig(PRES/'comparison.pdf');plt.close(fig)
 write(PRES/'verification.json',dict(passed=True,frames=decoded,seconds=37.8,fixed_keys=keys,selection='Unchanged original nine cases; seed17. Includes failures.'))
 print(dest,flush=True)
if __name__=='__main__':main()
