"""Numbered local mask videos for review; no annotation or training mutation."""
from audit import *
import sys,argparse
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder
cv2.setNumThreads(2)
DEST=OUT/'presentation'/VERSION

def render(c):
 cid=c['clip'];folder=OUT/cid;dest=DEST/'clips';dest.mkdir(parents=True,exist_ok=True)
 done=dest/f'{cid}_media.json'
 if done.exists():return
 a=read(folder/VERSION/'audit.json');rows=v.load(folder/VERSION/'masks_rle.jsonl.gz');raw=v.load(folder/'masks_rle.jsonl.gz')
 by=collections.defaultdict(list);counts=collections.Counter(r['frame'] for r in raw)
 for r in rows:by[r['frame']].append(r)
 ids=sorted({r['id'] for r in raw});cols={i:v.color(int(i[1:])-1) for i in ids}
 N=a['frames'];cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==N
 enc,log=encoder(dest/f'{cid}_masks.mp4',1280,1080,10,20)
 samples=sorted({0,N//2,N-1});saved=[]
 ww=[w for w in read(OUT/'selected_windows.json') if w['clip']==cid]
 for f in range(N):
  ok,frame=cap.read();assert ok,(cid,f)
  im,visible=v.overlay(frame,by[f],cols,{},f)
  canvas=np.zeros((1080,1280,3),np.uint8);canvas[:]=(23,20,18);canvas[82:1042]=im
  active=[w['window_id'] for w in ww if w['local_start_frame']<=f<=w['local_start_frame']+20]
  banner='场景暂不纳入 G1 近人标注；原因见索引' if a['scene_exclusion'] else ('本片段暂无可保留 mask；见待核查记录' if not rows else '近处优先 · SAM2 / V6 · 初筛结果，待你检查')
  lines=[(18,8,f'新增片段 {cid} / 118  |  原录制 {c["id"]}  |  {f/10:.1f}s  帧 {f:03d}'),(18,43,banner),(18,1049,f'窗口 {", ".join(active) or "—"}   当前显示 {len(visible)} 人   原始候选 {counts[f]} 个 mask')]
  canvas=v.text(canvas,lines,24);enc.stdin.write(canvas.tobytes())
  if f in samples:
   p=dest/f'{cid}_{f:03d}.jpg';cv2.imwrite(str(p),canvas);saved.append(str(p))
 cap.release();finish_encoder(enc,log)
 write(done,dict(clip=cid,frames=N,fps=10,duration_s=N/10,review_ids=a['review_ids'],review_masks=a['review_masks'],scene_exclusion=a['scene_exclusion'],samples=saved,video=str(dest/f'{cid}_masks.mp4')))
 print(cid,'RENDER',N,flush=True)

def compilations(clips):
 groups=[clips[i:i+29] for i in range(0,len(clips),29)]
 allenc,alllog=encoder(DEST/'ALL_118_CLIPS.mp4',1280,1080,10,21)
 posenc,poslog=encoder(DEST/'ALL_WITH_MASKS.mp4',1280,1080,10,21)
 timeline=[];offset=0;positive=0
 for gi,group in enumerate(groups,1):
  p=DEST/f'GROUP_{gi:02d}_{group[0]["clip"]}-{group[-1]["clip"]}.mp4';enc,log=encoder(p,1280,1080,10,21);local=0
  for c in group:
   cid=c['clip'];m=read(DEST/'clips'/f'{cid}_media.json');cap=cv2.VideoCapture(m['video']);n=0
   while True:
    ok,frame=cap.read()
    if not ok:break
    enc.stdin.write(frame.tobytes());allenc.stdin.write(frame.tobytes());n+=1
    if m['review_masks']:posenc.stdin.write(frame.tobytes())
   cap.release();assert n==m['frames']
   timeline.append(dict(clip=cid,all_start_s=offset/10,with_masks_start_s=positive/10 if m['review_masks'] else None,group=gi,group_start_s=local/10,frames=n,group_video=str(p)));offset+=n;local+=n
   if m['review_masks']:positive+=n
  finish_encoder(enc,log)
 finish_encoder(allenc,alllog);finish_encoder(posenc,poslog);write(DEST/'timeline.json',timeline)
 # Four clips per sheet, one first/middle/last strip per clip.
 sheets=DEST/'contact_sheets';sheets.mkdir(exist_ok=True)
 for j in range(0,len(clips),4):
  subset=clips[j:j+4];sheet=np.full((400*len(subset),1440,3),245,np.uint8)
  for k,c in enumerate(subset):
   m=read(DEST/'clips'/f'{c["clip"]}_media.json')
   for q,s in enumerate(m['samples']):sheet[k*400:(k+1)*400,q*480:(q+1)*480]=cv2.resize(cv2.imread(s),(480,400))
  cv2.imwrite(str(sheets/f'sheet_{j//4+1:02d}.jpg'),sheet)
 write(DEST/'complete.json',dict(clips=len(clips),frames=offset,duration_s=offset/10,groups=len(groups),with_masks_frames=positive,with_masks_duration_s=positive/10))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--clips',nargs='+');p.add_argument('--compile',action='store_true');a=p.parse_args()
 clips=read(OUT/'selected_clips.json')
 for c in clips:
  if a.clips and c['clip'] not in a.clips:continue
  if a.wait:
   while not(OUT/c['clip']/VERSION/'audit.json').exists():time.sleep(2)
  if (OUT/c['clip']/VERSION/'audit.json').exists():render(c)
 if a.compile and not(DEST/'complete.json').exists():compilations(clips)
