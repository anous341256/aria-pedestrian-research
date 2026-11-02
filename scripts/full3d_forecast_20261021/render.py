from geometry_common import *
import cv2,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
PRES=OUT/'presentation'
def main():
 PRES.mkdir(parents=True,exist_ok=True);d=dict(np.load(OUT/'dataset.npz'));meta=read(BASE/'samples.json');keys=read(ROOT/'outputs/biflow_aria_v1_20261002/presentation/selection.json')['fixed'];lookup={m['key']:i for i,m in enumerate(meta)};pred={}
 for fold in read(BASE/'folds.json'):
  old=np.load(BASE/'with_translation'/f'fold_{fold["fold"]}/seed_17/person/predictions.npz');new=np.load(OUT/f'fold_{fold["fold"]}/seed_17/person/predictions.npz');assert np.array_equal(old['indices'],new['indices'])
  for j,i in enumerate(new['indices']):pred[int(i)]={k:new[k][j] for k in new.files if k!='indices'};pred[int(i)]['old_image']=old['image'][j];pred[int(i)]['old_error']=old['image_error'][j]
 dest=PRES/'fixed_9_full3d_comparison.mp4';assert not dest.exists();ff='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';win='E:/'+str(dest).removeprefix('/mnt/e/');log=(PRES/'render.log').open('wb');proc=subprocess.Popen([ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x660','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',win],stdin=subprocess.PIPE,stderr=log)
 def txt(im,s,x,y,size=.65,c=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,2,cv2.LINE_AA)
 def dot(im,p,c,r=5):
  q=np.asarray(p)*.25
  if np.isfinite(q).all():cv2.circle(im,tuple(np.clip(q,-5000,5000).astype(int)),r,c,2,cv2.LINE_AA)
 count=0
 for key in keys:
  i=lookup[key];m=meta[i];p=pred[i];cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][0]);frames=[]
  for j in range(21):
   ok,im=cap.read();assert ok;frames.append(cv2.resize(im,(640,480)))
  cap.release()
  for j,im in enumerate(frames):
   out=np.zeros((660,1920,3),np.uint8);txt(out,f'{key} | Fixed original cases | Seed17 | History 1.5s -> future 0.5s',14,30,.8)
   for col,(name,arr,err) in enumerate([('Previous translation model',p['old_image'],p['old_error']),('Full 3D + predicted camera',p['image'],p['image_error']),('Diagnostic: same 3D + GT camera',p['gt_camera'],p['gt_camera_error'])]):
    panel=im.copy();dot(panel,d['xy'][i,j],(255,255,255))
    if j<=15:
     for t in range(j+1):dot(panel,d['xy'][i,t],(255,200,60),3)
    else:
     point=arr[j-16];dot(panel,point,(0,160,255),9)
     if np.isfinite(point).all():cv2.line(panel,tuple(np.clip(d['xy'][i,j]*.25,-5000,5000).astype(int)),tuple(np.clip(point*.25,-5000,5000).astype(int)),(0,160,255),2,cv2.LINE_AA)
    out[88:568,col*640:(col+1)*640]=panel;txt(out,name+f' | ADE {err.mean():.1f}px',col*640+10,65,.55)
   phase=f'Observed history {(j-15)*.1:+.1f}s' if j<=15 else f'Fixed forecast +{(j-15)*.1:.1f}s; no new observations'
   txt(out,phase+' | White: reference center; orange: prediction',14,600,.72);txt(out,'Right panel uses future reference camera ONLY for diagnosis. Main forecast uses predicted rotation + translation.',14,638,.65)
   proc.stdin.write(out.tobytes());proc.stdin.write(out.tobytes());count+=2
   if j==20:cv2.imwrite(str(PRES/f'{key}.jpg'),out)
  fig,axs=plt.subplots(2,2,figsize=(12,8),dpi=160);history=np.arange(-15,1)/10;future=np.arange(1,6)/10;valid=d['depth_valid'][i];xyz=d['xyz'][i].copy();xyz[~valid]=np.nan
  ax=axs[0,0];ax.plot(d['xy'][i,:16,0],d['xy'][i,:16,1],'-',color='#577590',label='Observed history');ax.plot(d['xy'][i,15:,0],d['xy'][i,15:,1],'-o',color='#43aa8b',label='Future reference');ax.plot(p['image'][:,0],p['image'][:,1],'-o',color='#f8961e',label='Full3D forecast');ax.set_title('Final image trajectory');ax.set_xlabel('u (px)');ax.set_ylabel('v (px)');ax.invert_yaxis();ax.legend(fontsize=8)
  ax=axs[0,1];ax.plot(xyz[:16,0],xyz[:16,2],'-o',color='#577590',ms=3,label='Compensated history');ax.plot(xyz[15:,0],xyz[15:,2],'-o',color='#43aa8b',label='Future depth proxy');ax.plot(p['person_xyz'][:,0],p['person_xyz'][:,2],'-o',color='#f8961e',label='Person head');ax.set_title('Person branch: fixed-frame X-Z view');ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)');ax.legend(fontsize=8)
  rv=Rotation.from_matrix(d['A'][i,16:]).as_rotvec();colors=['#577590','#43aa8b','#f8961e']
  for k,(label,color) in enumerate(zip('XYZ',colors)):
   axs[1,0].plot(future,np.rad2deg(rv[:,k]),color=color,label=label+' ref');axs[1,0].plot(future,np.rad2deg(p['rotation'][:,k]),'--',color=color,label=label+' pred');axs[1,1].plot(future,d['translation'][i,16:,k],color=color,label=label+' ref');axs[1,1].plot(future,p['translation'][:,k],'--',color=color,label=label+' pred')
  axs[1,0].set_title('Camera branch: relative rotation vector');axs[1,0].set_ylabel('Degrees');axs[1,1].set_title('Camera branch: relative translation');axs[1,1].set_ylabel('Meters')
  for ax in axs[1]:ax.set_xlabel('Future time (s)');ax.legend(ncol=3,fontsize=8)
  for ax in axs.ravel():ax.grid(alpha=.2)
  fig.suptitle(key+' | Separate outputs and composition | 3D reference is a depth-based proxy');fig.tight_layout();fig.savefig(PRES/f'{key}_branches.png');plt.close(fig)
 proc.stdin.close();assert proc.wait(timeout=60)==0;log.close();cap=cv2.VideoCapture(str(dest));decoded=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape[:2]==(660,1920);decoded+=1
 cap.release();assert decoded==count==378
 r=read(OUT/'comparison.json')['results'];names=['saved_translation','full3d','fixed3D','CV3D','hold_last'];labels=['Previous translation','Full3D model','Fixed3D + camera','CV3D + camera','Hold image'];fig,axs=plt.subplots(1,2,figsize=(12,4),dpi=160)
 for ax,key in zip(axs,['ADE','FDE']):
  vals=[r[n]['window_mean'][key] for n in names];ax.bar(labels,vals,color=['#577590','#43aa8b','#90be6d','#f8961e','#999999']);ax.set_ylabel(key+' (pixels)');ax.tick_params(axis='x',rotation=20);ax.set_ylim(0,max(vals)*1.18)
  for j,v in enumerate(vals):ax.text(j,v+1,f'{v:.1f}',ha='center',fontsize=8)
 fig.suptitle('Full 3D compensation and prediction | 117 windows, 6 recording folds, 3 seeds');fig.tight_layout();fig.savefig(PRES/'comparison.png');fig.savefig(PRES/'comparison.pdf');plt.close(fig);write(PRES/'verification.json',dict(passed=True,frames=decoded,seconds=37.8,fixed_keys=keys,selection='Original nine keys and seed17, no score-based replacements'));print(dest)
if __name__=='__main__':main()
