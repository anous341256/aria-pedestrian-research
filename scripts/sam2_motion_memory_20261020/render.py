"""Four-panel raw propagation and final annotation videos; no quality claims."""
import sys,json,gzip,hashlib,subprocess,collections,argparse
from pathlib import Path
import cv2,numpy as np
from cvat_sdk.masks import decode_mask
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_motion_memory_20261020'
ARMS=['visual','external_box','camera','camera_person']
TITLES=['A  Visual memory','B  External motion box','C  Internal camera prior','D  Internal camera + target prior']
FFMPEG='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def load(p):
 with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def decode(rle):
 # CVAT RLE includes tight box in final four coordinates; reuse SDK contract.
 return decode_mask(rle,image_width=2560,image_height=1920)
def win(p):return 'E:'+str(p).split('/mnt/e',1)[1]
def encoder(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 return subprocess.Popen([FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
  '-pixel_format','bgr24','-video_size','1280x1056','-framerate','5','-i','pipe:0',
  '-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',win(path)],stdin=subprocess.PIPE)
def color(identity):
 return tuple(int(x) for x in np.random.default_rng(int(hashlib.md5(identity.encode()).hexdigest()[:8],16)).integers(70,240,3))
def overlay(im,rows,statuses,frame):
 out=cv2.resize(im,(640,480))
 for r in rows:
  if r['frame']!=frame:continue
  m=decode(r['rle']);m=cv2.resize(m.astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)
  col=color(r['id']);out[m]=(out[m]*.55+np.array(col)*.45).astype(np.uint8)
  cont,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(out,cont,-1,col,1)
  y,x=np.nonzero(m)
  if len(x):
   label=r['id'];status=statuses.get((r['segment'],frame),'')
   if 'REPAIR' in status:label+=' R'
   elif 'UNCERTAIN' in status:label+=' ?'
   cv2.putText(out,label,(int(np.median(x)),max(14,int(y.min())-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
   cv2.putText(out,label,(int(np.median(x)),max(14,int(y.min())-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1,cv2.LINE_AA)
 return out
def render_clip(c,paths=None):
 dest=OUT/c['id']
 if not all((dest/a/'complete.json').exists() for a in ARMS):return None
 pres=OUT/'presentation';paths=paths or {v:pres/v/(c['id']+'.mp4') for v in ['raw','final']}
 if all(p.exists() for p in paths.values()):return paths
 data={};status={}
 for arm in ARMS:
  final=load(dest/arm/'masks_rle.jsonl.gz');logs=load(dest/arm/'decisions.jsonl.gz')
  raw=[dict(id=x['id'],segment=x['segment'],frame=x['frame'],rle=x['raw_rle']) for x in logs if 'raw_rle' in x]
  raw.extend(x for x in final if any(l['segment']==x['segment'] and l['frame']==x['frame'] and l['status']=='ANCHOR' for l in logs))
  data[arm]=dict(raw=raw,final=final)
  status[arm]={(x['segment'],x['frame']):x['status'] for x in logs}
 cap=cv2.VideoCapture(str(ROOT/c['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,c['lo'])
 procs={v:encoder(p) for v,p in paths.items()}
 for f in range(c['lo'],c['hi']):
  ok,im=cap.read();assert ok
  for variant,proc in procs.items():
   canvas=np.full((1056,1280,3),245,np.uint8)
   top=f"Clip {c['id']} | {c['recording'][-6:]} | source {(c['start']+f)/10:.1f}s | {'BEFORE proposal repair' if variant=='raw' else 'FINAL annotation'}"
   cv2.putText(canvas,top,(16,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA)
   cv2.putText(canvas,'0.5x playback | Same initial masks | R: repair triggered | ?: uncertain | NOT manual ground truth',(16,48),cv2.FONT_HERSHEY_SIMPLEX,.49,(80,80,80),1,cv2.LINE_AA)
   for index,arm in enumerate(ARMS):
    x=(index%2)*640;y=60+(index//2)*498
    cv2.putText(canvas,TITLES[index],(x+10,y+15),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
    canvas[y+18:y+498,x:x+640]=overlay(im,data[arm][variant],status[arm],f)
   proc.stdin.write(canvas.tobytes())
 cap.release()
 for p in procs.values():p.stdin.close();assert p.wait()==0
 return paths
def concat(paths,dest):
 listing=dest.with_suffix('.concat.txt');listing.parent.mkdir(parents=True,exist_ok=True)
 listing.write_text('\n'.join("file '"+win(p).replace("'","'\\''")+"'" for p in paths),encoding='utf8')
 subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',win(listing),'-c','copy','-movflags','+faststart',win(dest)],check=True)
def verify(path):
 cap=cv2.VideoCapture(str(path));count=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  count+=1
 fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
 return dict(file=str(path.relative_to(ROOT)),frames=count,fps=fps,seconds=count/fps if fps else None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='*');a=p.parse_args()
 selected=read(OUT/'selection.json')
 if a.clips:selected=[x for x in selected if x['id'] in a.clips]
 for c in selected:
  render_clip(c);print('Rendered',c['id'],flush=True)
if __name__=='__main__':main()
