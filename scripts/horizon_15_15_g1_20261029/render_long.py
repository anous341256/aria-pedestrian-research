"""Fixed first case per recording, plus long G1 target review sheets."""
from pathlib import Path
import json,subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/horizon_15_15_g1_20261029';PRES=OUT/'presentation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def frame(m,j,cache):
 if m['training_source']=='old_reviewed_union':path=ROOT/m['input']/'rgb.mp4';f=m['frames'][j]
 else:
  s=m['source_frames'][j];path=ROOT/s['input']/'rgb.mp4';f=s['frame']
 key=str(path)
 if key not in cache:
  cap=cv2.VideoCapture(key);imgs=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   imgs.append(im)
  cap.release();cache[key]=imgs
 assert f<len(cache[key]);return cache[key][f].copy()
def text(im,s,x,y,scale=.6,color=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def dot(im,p,color,r=4):
 if np.isfinite(p).all() and abs(p).max()<1e6:cv2.circle(im,tuple(np.rint(p).astype(int)),r,color,-1,cv2.LINE_AA)
def poly(im,points,color):
 if len(points)>1 and np.isfinite(points).all() and abs(points).max()<1e6:cv2.polylines(im,[np.rint(points).astype(np.int32)],False,color,2,cv2.LINE_AA)
def main():
 PRES.mkdir(parents=True,exist_ok=True);d=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cfg=read(OUT/'protocol.json');folds=read(OUT/'folds.json');cache={};selected=[next(i for i,m in enumerate(meta[:cfg['original_windows']]) if m['recording']==g) for g in cfg['test_recordings']];pred={};foldof={i:f['fold'] for f in folds for i in f['test_indices']}
 for f in folds:
  for a in cfg['arms']:
   z=dict(np.load(OUT/a/f'fold_{f["fold"]}/seed_17/predictions.npz'))
   for j,i in enumerate(z['indices']):pred[(a,int(i))]=dict(image=z['image'][j],ADE=float(z['image_error'][j].mean()))
 colors=[(210,210,210),(80,160,255),(80,230,100)];dest=PRES/'fixed_5_long_horizon_comparison.mp4';temp=PRES/'render_temp.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),10,(1440,430));count=0;finals=[]
 for i in selected:
  m=meta[i];hold=np.repeat(d['xy'][i,15,None],15,0);ps=[hold,pred[('original_long',i)]['image'],pred[('expanded_long',i)]['image']];errs=[float(np.linalg.norm(p-d['xy'][i,16:31],axis=-1).mean()) for p in ps]
  for j in range(31):
   rgb=frame(m,j,cache);panels=[]
   for a,(label,p,e,col) in enumerate(zip(['Hold last position','Dual branch: old data','Dual branch: + new G1'],ps,errs,colors)):
    im=np.zeros((430,480,3),np.uint8);im[55:415]=cv2.resize(rgb,(480,360));sc=np.array([480/2560,360/1920]);off=np.array([0,55]);text(im,label,12,22,.55,col);text(im,f'{m["key"]}  ADE {e:.1f}px',12,44,.46);hist=d['xy'][i,:min(j+1,16)]*sc+off;poly(im,hist,(220,220,220));dot(im,d['xy'][i,j]*sc+off,(255,255,255),4)
    if j>=15:
     poly(im,np.vstack([d['xy'][i,15],p])*sc+off,col)
     if j>15:poly(im,d['xy'][i,15:j+1]*sc+off,(255,210,30));dot(im,p[j-16]*sc+off,col,5)
    b=d['boxes'][i,j].reshape(2,2)*sc+off;cv2.rectangle(im,tuple(b[0].astype(int)),tuple(b[1].astype(int)),(180,180,180),1);text(im,f't={(j-15)/10:+.1f}s | white: reference | colored: prediction',10,427,.36);panels.append(im)
   canvas=np.hstack(panels);writer.write(canvas);count+=1
  cv2.imwrite(str(PRES/f'case_{i:03d}_last.jpg'),canvas);finals.append(canvas)
 writer.release();subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True);temp.unlink();cap=cv2.VideoCapture(str(dest));decoded=0
 while True:
  ok,_=cap.read()
  if not ok:break
  decoded+=1
 cap.release();assert decoded==count;cv2.imwrite(str(PRES/'fixed_5_final_frames.jpg'),np.vstack(finals))
 # New-data samples selected before looking at model errors: first per recording.
 newfirst=[]
 for g in sorted({m['recording'] for m in meta[cfg['original_windows']:]}):newfirst.append(next(i for i,m in enumerate(meta) if i>=cfg['original_windows'] and m['recording']==g))
 review=[]
 for i in newfirst:
  cells=[]
  for j in [0,15,30]:
   im=frame(meta[i],j,cache);b=d['boxes'][i,j].astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,0),5);im=cv2.resize(im,(480,360));cv2.rectangle(im,(0,0),(480,42),(15,15,15),-1);text(im,f'{meta[i]["key"]}  frame {j}',10,18,.45);text(im,meta[i]['recording'],10,36,.40);cells.append(im)
  review.append(np.hstack(cells))
 for k in range(0,len(review),4):cv2.imwrite(str(PRES/f'new_G1_review_{k//4+1:02d}.jpg'),np.vstack(review[k:k+4]))
 c=read(OUT/'comparison.json');fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');labels=['Old data','+ new G1','Hold last'];arms=cfg['arms']+['hold_last'];cols=['#d58c36','#278456','#8b929a']
 for ax,metric in zip(axs,['ADE','FDE']):
  vals=[c['results'][a]['window_mean'][metric] for a in arms];ax.bar(labels,vals,color=cols);ax.set_ylabel(metric+' (pixels)');ax.set_title('Future 1.5 seconds');ax.spines[['top','right']].set_visible(False)
  for j,v in enumerate(vals):ax.text(j,v,f'{v:.1f}',ha='center',va='bottom')
 fig.savefig(PRES/'comparison.png',dpi=180);plt.close(fig)
 rows=read(OUT/'metrics.json');write(PRES/'visualization_manifest.json',dict(selected_test_indices=selected,selection='First old eligible window per recording, seed17. No error-based selection.',new_review_indices=newfirst,video_frames=decoded,video_seconds=decoded/10));(PRES/'START_HERE.txt').write_text('先看 comparison.png，再看 fixed_5_long_horizon_comparison.mp4。\n视频为每个测试录制固定第一例、seed17；总表为三个种子平均。\n左保持位置，中旧数据，右加新G1；白点参考中心，青色参考未来轨迹，彩色预测。\n新G1抽样见new_G1_review_*.jpg，每个新增录制固定第一窗的起点/预测起点/终点。\n',encoding='utf-8');print(dict(video_frames=decoded,cases=len(selected),new_review_windows=len(newfirst)),flush=True)
if __name__=='__main__':main()
