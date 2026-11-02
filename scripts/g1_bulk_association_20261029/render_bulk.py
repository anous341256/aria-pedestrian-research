from bulk_common import *
import cv2,subprocess
cv2.setNumThreads(1)
PRES=OUT/'presentation'
def text(im,s,x,y,size=.6,color=(240,240,240)):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
def color(tid):return tuple(int(x) for x in np.random.default_rng(tid).integers(60,250,3))
def main():
 PRES.mkdir(parents=True,exist_ok=True);inv=read(OUT/'inventory.json');start=time.monotonic();allframes=0;overview=[];overviewkeys=[];targetcards=[];cardkeys=[];clips=[]
 for cl in inv['clips']:
  cid=cl['clip'];out=PRES/'clips'/f'{cid}.mp4';out.parent.mkdir(parents=True,exist_ok=True)
  if not (OUT/'quality'/f'{cid}.json').exists():continue
  if out.exists() and (out.with_suffix('.json')).exists():continue
  entries=[e for e in inv['entries'] if e['clip']==cid];target={t['target_key']:t for e in entries for t in e['targets']};old=read(OUT/'tracking_baseline'/f'{cid}.json')['rows'];new=read(OUT/'tracking'/f'{cid}.json')['rows'];qual=read(OUT/'quality'/f'{cid}.json');cap=cv2.VideoCapture(str(ROOT/cl['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,cl['lo']);frames=[];snapshot=[]
  ff='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';args=[ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x560','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','E:/'+str(out).removeprefix('/mnt/e/')];log=out.with_suffix('.log').open('wb');proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
  for f,e in enumerate(entries):
   ok,im=cap.read();assert ok;frames.append(im);canvas=np.zeros((560,1280,3),np.uint8)
   for j,(rows,label) in enumerate([(old,'Global appearance + geometry'),(new,'Local + global appearance + geometry')]):
    panel=cv2.resize(im,(640,480))
    for r in rows:
     if r['frame']!=f:continue
     mask=cv2.resize(decode(target[r['uid']]['rle']),(640,480),interpolation=cv2.INTER_NEAREST)>0;col=color(r['track_id']);panel[mask]=(.65*panel[mask]+.35*np.array(col)).astype(np.uint8);b=(np.array(target[r['uid']]['box'])*.25).astype(int);cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),col,1);text(panel,f'T{r["track_id"]}',int(b[0]),max(15,int(b[1])-4),.5,col)
     if j and (qual['frame_audit'][r['uid']]['reasons'] or qual['frame_audit'][r['uid']]['manual_review_flags']):text(panel,'?',int(b[2])-10,max(15,int(b[1])-4),.6,(0,165,255))
    canvas[50:530,j*640:(j+1)*640]=panel;text(canvas,label,j*640+10,40,.58)
   text(canvas,f'G1 {cid} | {cl["recording"][-6:]} | frame {f:02d}/20 | Same SAM masks; IDs reassociated',10,18,.57);text(canvas,'Colors = output track IDs (clip-local).  ? = automatic quality/review flag; masks retained.',10,551,.57);proc.stdin.write(canvas.tobytes())
   if f==10:snapshot=canvas.copy();cv2.imwrite(str(PRES/f'{cid}_mid.jpg'),canvas)
  cap.release();proc.stdin.close();assert proc.wait(timeout=60)==0;log.close();write(out.with_suffix('.json'),dict(frames=len(entries),width=1280,height=560));print('VIDEO',cid,flush=True)
  # All full-window target identities, start/middle/end, for visual inspection.
  for w in qual['windows']:
   strip=np.zeros((300,900,3),np.uint8)
   for j,k in enumerate([0,10,20]):
    u=w['uids'][k];b=np.array(target[u]['box']);im=frames[w['frames'][k]];pad=.2*max(b[2]-b[0],b[3]-b[1]);x1,y1=np.maximum(np.floor(b[:2]-pad),0).astype(int);x2,y2=np.minimum(np.ceil(b[2:]+pad),[2560,1920]).astype(int);crop=im[y1:y2,x1:x2].copy();mask=decode(target[u]['rle'])[y1:y2,x1:x2]>0;crop[mask]=(.7*crop[mask]+.3*np.array(color(w['track_id']))).astype(np.uint8);s=min(300/crop.shape[1],265/crop.shape[0]);crop=cv2.resize(crop,(max(1,round(crop.shape[1]*s)),max(1,round(crop.shape[0]*s))));strip[30:30+crop.shape[0],j*300:j*300+crop.shape[1]]=crop
   text(strip,w['key']+' | '+('QUALITY PASS' if w['quality_eligible'] else 'REVIEW FLAG'),8,20,.55);dest=PRES/'target_strips'/f'{w["key"]}.jpg';dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),strip)
 if any(not (PRES/'clips'/f'{c["clip"]}.json').exists() for c in inv['clips']):
  print('READY VIDEO EXPORT COMPLETE; remaining clips will resume after association.',flush=True);return
 # Assemble paginated interval overviews and target strips from all files (resumable).
 for cl in inv['clips']:overview.append(cv2.resize(cv2.imread(str(PRES/f'{cl["clip"]}_mid.jpg')),(640,280)));overviewkeys.append(cl['clip'])
 for name,images,keys,cols,rows in [('overview',overview,overviewkeys,2,4)]:
  for page,startidx in enumerate(range(0,len(images),cols*rows),1):
   cells=images[startidx:startidx+cols*rows];h,w=cells[0].shape[:2];canvas=np.zeros((h*rows,w*cols,3),np.uint8)
   for j,im in enumerate(cells):canvas[(j//cols)*h:(j//cols+1)*h,(j%cols)*w:(j%cols+1)*w]=im
   cv2.imwrite(str(PRES/f'{name}_{page:02d}.jpg'),canvas)
 strips=sorted((PRES/'target_strips').glob('*.jpg'))
 expected_windows=read(OUT/'forecast_window_candidates.json')['windows'];assert len(strips)==len(expected_windows)
 for page,begin in enumerate(range(0,len(strips),8),1):
  canvas=np.zeros((1200,1800,3),np.uint8)
  for j,p in enumerate(strips[begin:begin+8]):canvas[(j//2)*300:(j//2+1)*300,(j%2)*900:(j%2+1)*900]=cv2.imread(str(p))
  cv2.imwrite(str(PRES/f'targets_{page:02d}.jpg'),canvas)
 listing=PRES/'concat.txt';listing.write_text('\n'.join("file '"+str(PRES/'clips'/f'{c["clip"]}.mp4').replace('/mnt/e/','E:/')+"'" for c in inv['clips']),encoding='utf-8');dest=PRES/'all191_association_comparison.mp4'
 if not dest.exists():subprocess.run([ff,'-v','error','-n','-f','concat','-safe','0','-i','E:/'+str(listing).removeprefix('/mnt/e/'),'-c','copy','-movflags','+faststart','E:/'+str(dest).removeprefix('/mnt/e/')],check=True)
 for p in [dest]+sorted((PRES/'clips').glob('*.mp4')):
  cap=cv2.VideoCapture(str(p));n=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape[:2]==(560,1280);n+=1
  cap.release();assert n==(4011 if p==dest else 21);allframes+=n
 write(PRES/'verification.json',dict(passed=True,decoded_frames=allframes,video_files=192,overview_pages=24,target_strips=len(strips),target_pages=(len(strips)+7)//8,seconds=time.monotonic()-start));print('RENDER DONE',allframes,flush=True)
if __name__=='__main__':main()
