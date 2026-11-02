"""Preserve the original nine showcase cases and seed; all outcomes retained."""
from forecast_common import *
import subprocess,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PRES=OUT/'presentation'

def main():
 PRES.mkdir(parents=True,exist_ok=True);d=np.load(BASE/'dataset.npz');meta=read(BASE/'samples.json');keys=read(ROOT/'outputs/biflow_aria_v1_20261002/presentation/selection.json')['fixed'];lookup={m['key']:i for i,m in enumerate(meta)};pred={};arms=['saved_translation']+ARMS;names=['Saved original','Original-data fine-tune','Trimmed-data fine-tune']
 for fold in read(OUT/'folds.json'):
  for arm in arms:
   path=BASE/'with_translation'/f'fold_{fold["fold"]}/seed_17/person/predictions.npz' if arm==arms[0] else OUT/arm/f'fold_{fold["fold"]}/seed_17/predictions.npz'
   z=np.load(path)
   for j,i in enumerate(z['indices']):pred.setdefault(int(i),{})[arm]=(z['image'][j],z['image_error'][j].mean())
 dest=PRES/'fixed_9_trimmed_finetune_comparison.mp4';ffmpeg='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';args=[ffmpeg,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x620','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','E:/'+str(dest).removeprefix('/mnt/e/')]
 assert not dest.exists();log=(PRES/'render.log').open('wb');proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
 def txt(im,s,x,y,size=.7,color=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)
 def pt(im,p,color,r=7):
  q=np.asarray(p)*.25
  if np.isfinite(q).all():cv2.circle(im,tuple(np.clip(q,[-1000,-1000],[1640,1480]).astype(int)),r,color,2,cv2.LINE_AA)
 count=0;cards=[];caseinfo=[]
 for key in keys:
  i=lookup[key];m=meta[i];cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'));assert cap.isOpened();frames=[];cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][0])
  for j in range(21):
   ok,im=cap.read();assert ok;frames.append(cv2.resize(im,(640,480)))
  cap.release();truth=d['xy'][i,16:21];caseinfo.append(dict(key=key,seed=17,ADE={a:float(pred[i][a][1]) for a in arms}))
  for j,im in enumerate(frames):
   out=np.zeros((620,1920,3),np.uint8);txt(out,f'{key} | Same held-out test case | Frozen camera | Seed17 | Half speed',14,30,.8)
   for col,(arm,name) in enumerate(zip(arms,names)):
    arr,ade=pred[i][arm];panel=im.copy();pt(panel,d['xy'][i,j],(255,255,255),6)
    if j<=15:
     for t in range(j+1):pt(panel,d['xy'][i,t],(255,200,60),3)
    else:
     pt(panel,arr[j-16],(0,160,255),9);a=np.clip(truth[j-16]*.25,-5000,5000).astype(int);b=np.clip(arr[j-16]*.25,-5000,5000).astype(int);cv2.line(panel,tuple(a),tuple(b),(0,160,255),2,cv2.LINE_AA)
    out[88:568,col*640:(col+1)*640]=panel;txt(out,name+f' | ADE {ade:.1f}px',col*640+10,65,.63)
   phase=f'History: t={(j-15)*.1:+.1f}s' if j<=15 else f'Fixed forecast: +{(j-15)*.1:.1f}s (no new observations)';txt(out,phase+' | White: SAM reference center. Orange: forecast.',14,605,.76)
   proc.stdin.write(out.tobytes());proc.stdin.write(out.tobytes());count+=2
   if j==20:cv2.imwrite(str(PRES/f'{key}.jpg'),out);cards.append(cv2.resize(out,(960,310)))
 proc.stdin.close();assert proc.wait(timeout=60)==0;log.close();cv2.imwrite(str(PRES/'fixed_9_contact_sheet.jpg'),np.concatenate(cards,axis=0));cap=cv2.VideoCapture(str(dest));decoded=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape[:2]==(620,1920);decoded+=1
 cap.release();assert decoded==count==378;write(PRES/'fixed_9_cases.json',caseinfo)
 c=read(OUT/'comparison.json');fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');colors=['#64748b','#3b82f6','#059669','#a1a1aa'];aa=arms+['hold_last'];labels=['Saved','Original-data\nfine-tune','Trimmed-data\nfine-tune','Hold last']
 for axis,metric in zip(ax,['ADE','FDE']):
  vals=[c['results'][a]['window_mean'][metric] for a in aa];bars=axis.bar(labels,vals,color=colors,width=.65);axis.bar_label(bars,fmt='%.2f',padding=4);axis.set_ylim(0,max(vals)*1.2);axis.set_ylabel(metric+' (pixels; lower is better)');axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
 fig.suptitle('Same 117 held-out windows / 6 recordings / 3 seeds\n1.5 s history → 0.5 s forecast; saved camera frozen',fontsize=12);fig.savefig(PRES/'comparison.png',dpi=180);plt.close(fig)
 write(PRES/'verification.json',dict(passed=True,frames=decoded,seconds=37.8,fixed_keys=keys,selection='Same original nine cases fixed before this experiment; seed17; no score-based replacement.'))
 print(str(dest),flush=True)
if __name__=='__main__':main()
