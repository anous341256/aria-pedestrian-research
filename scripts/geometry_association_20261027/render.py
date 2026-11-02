from assoc_common import *
import cv2,subprocess,collections
PRES=OUT/'presentation'
NAMES=['Image motion','Rotation compensation','Rotation + translation + depth']
def txt(im,s,x,y,size=.6,color=(235,235,235)):
 cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(15,15,15),4,cv2.LINE_AA);cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
def color(tid):
 h=np.uint8([[[int(tid*47%180),200,255]]]);return tuple(map(int,cv2.cvtColor(h,cv2.COLOR_HSV2BGR)[0,0]))
def encoder(path):
 ff='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe';log=path.with_suffix('.log').open('wb');p=subprocess.Popen([ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x640','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','E:/'+str(path).removeprefix('/mnt/e/')],stdin=subprocess.PIPE,stderr=log);return p,log
def finish(p,l):p.stdin.close();assert p.wait(timeout=60)==0;l.close()
def main():
 PRES.mkdir(exist_ok=True);inv=read(OUT/'inventory.json');ref=read(OUT/'source_identity_reference.json');comp=read(OUT/'comparison.json');records=[];snapshots=[]
 for cond in ['natural','blackout3']:
  dest=PRES/f'all10_{cond}_comparison.mp4';assert not dest.exists();enc,log=encoder(dest)
  for cl in inv['clips']:
   cid=cl['clip'];entries=[e for e in inv['entries'] if e['clip']==cid];rr={a:read(OUT/'tracking'/cond/a/f'{cid}.json')['rows'] for a in ARMS};lookup={a:{r['uid']:r for r in rows} for a,rows in rr.items()};changed={a:{r['uid'] for r in comp[cond][a]['per_clip'][cid]['changes']} for a in ARMS};cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));assert cap.isOpened();single=PRES/f'{cid}_{cond}.mp4';ce,clog=encoder(single)
   for e in entries:
    ok,im=cap.read();assert ok;base=cv2.resize(im,(640,480));out=np.full((640,1920,3),20,np.uint8);f=e['frame'];hidden=cond=='blackout3' and f>=20 and f%20<3
    txt(out,f'Clip {cid} | Frame {f:02d} | {cond} | SAME frozen SAM2 candidates; IDs assigned independently',15,28,.78)
    for col,a in enumerate(ARMS):
     panel=base.copy();records_here=[]
     for t in e['targets']:
      uid=t['target_key'];r=lookup[a].get(uid);b=np.asarray(t['box'])/4;mask=cv2.resize(decode(t['rle']),(640,480),interpolation=cv2.INTER_NEAREST)>0
      if r is None:continue
      rgb=color(r['track_id']);panel[mask]=(.78*panel[mask]+.22*np.array(rgb)).astype(np.uint8);x1,y1,x2,y2=np.clip(b,[0,0,0,0],[639,479,639,479]).astype(int);cv2.rectangle(panel,(x1,y1),(x2,y2),rgb,2);sid=ref[uid]['source_id'];label=f'T{r["track_id"]} / ref {sid}';label+=' *' if uid in changed[a] else '';txt(panel,label,max(1,x1),max(16,y1-5),.48,(0,100,255) if uid in changed[a] else rgb)
      if 'predicted_uv' in r:
       uv=np.array(r['predicted_uv'])/4
       if np.isfinite(uv).all() and (abs(uv)<5000).all():
        q=tuple(uv.astype(int));cv2.drawMarker(panel,q,(0,180,255),cv2.MARKER_CROSS,14,2)
        C=np.array(r['prior_cov'])/16;vals,vec=np.linalg.eigh(C);axes=tuple(np.clip(np.sqrt(np.maximum(vals,0)*5.991),1,180).astype(int));angle=float(np.degrees(np.arctan2(vec[1,0],vec[0,0])));cv2.ellipse(panel,q,axes,angle,0,360,(0,180,255),1,cv2.LINE_AA)
     out[100:580,col*640:(col+1)*640]=panel;stats=comp[cond][a]['per_clip'][cid];txt(out,NAMES[col],col*640+12,60,.68);txt(out,f'Ref-ID changes: {stats["source_id_assignment_changes"]} | output tracks: {stats["output_tracks"]}',col*640+12,84,.52)
    if hidden:txt(out,'SIMULATED MISSING DETECTIONS: candidates, appearance and depth withheld this frame',15,610,.74,(0,180,255))
    else:txt(out,'Orange + / ellipse: predicted position / heuristic 95% region. *: output ID changed for a source reference.',15,610,.65)
    enc.stdin.write(out.tobytes());ce.stdin.write(out.tobytes())
    if cond=='natural' and f==cl['frames']//2:cv2.imwrite(str(PRES/f'{cid}_midframe.jpg'),out);snapshots.append(f'{cid}_midframe.jpg')
   finish(ce,clog);cap.release();records.append((single,cl['frames']));print('RENDER',cond,cid,flush=True)
  finish(enc,log);records.append((dest,530))
 for path,n in records:
  cap=cv2.VideoCapture(str(path));j=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape[:2]==(640,1920);j+=1
  cap.release();assert j==n,(path,j,n)
 write(PRES/'verification.json',dict(passed=True,videos=len(records),decoded_frames=sum(n for _,n in records),each_montage_seconds=53.,all10_clips_included=True,fixed_midframes=snapshots))
if __name__=='__main__':main()
