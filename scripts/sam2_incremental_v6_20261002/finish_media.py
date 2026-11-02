"""Assemble full-length collections, cumulative batches, and verify all decoded frames."""
import subprocess,collections
import cv2,numpy as np
from render import *
from common import ffmpeg,winpath
DEST=OUT/'presentation'
def join(kind):
    target=DEST/f'all_10_{kind}.mp4'
    if target.exists():return
    listing=DEST/f'concat_{kind}.txt';listing.write_text('\n'.join("file '"+winpath(DEST/f'{i:02d}_{kind}.mp4').replace('\\','/')+"'" for i in range(1,11)),encoding='utf8')
    subprocess.run([str(ffmpeg()),'-hide_banner','-loglevel','error','-nostdin','-f','concat','-safe','0','-i',winpath(listing),'-c','copy','-movflags','+faststart','-n',winpath(target)],check=True)
def stages(cid):
    target=DEST/f'{cid}_batch_growth.mp4'
    if target.exists():return
    inv=read_json(OUT/cid/'inventory.json');rows=load(OUT/cid/'masks_rle.jsonl.gz');by=collections.defaultdict(list)
    for r in rows:by[r['frame']].append(r)
    cols={e['id']:color(j) for j,e in enumerate(inv['entries'])};N=read_json(SRC/cid/'input/metadata.json')['frames']
    # Repeat the full clip once per completed batch, rather than showing a single favorable frame.
    enc,log=encoder(target,1280,1060,10,20)
    for b in range(1,inv['batches']+1):
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));ids=[]
        for p in sorted((OUT/cid).glob('batch_*/input.json')):
            data=read_json(p)
            if data['batch']<=b:ids.extend(e['id'] for e in data['entries'])
        for f in range(N):
            ok,frame=cap.read();assert ok;vis,present=overlay(frame,[r for r in by[f] if r['batch']<=b],cols,{},f)
            canvas=np.zeros((1060,1280,3),np.uint8);canvas[60:1020]=vis
            canvas=text(canvas,[(18,8,f'{cid} 分批累加 | 完成第 {b}/{inv["batches"]} 批 | 累计 {len(ids)} 个候选 ID'),(18,1027,f'整段重放：{f/10:.1f} 秒 · 当前 {len(present)} 个 mask · 已有批次输出保持原样')],25);enc.stdin.write(canvas.tobytes())
        cap.release()
    finish_encoder(enc,log)
def verify():
    results=[]
    for p in sorted(DEST.glob('*.mp4')):
        cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(h,w);n+=1
        cap.release()
        if p.name.startswith('all_10'):expected=530
        else:
            cid=p.name[:2];expected=read_json(SRC/cid/'input/metadata.json')['frames']
            if 'batch_growth' in p.name:expected*=read_json(OUT/cid/'inventory.json')['batches']
        assert n==expected and fps==10,(p,n,expected,fps);results.append(dict(file=p.name,frames=n,fps=fps,width=w,height=h,bytes=p.stat().st_size))
    write_json(DEST/'verification.json',dict(status='all_complete_videos_decode',videos=results));print('verified',len(results),'videos',sum(r['frames'] for r in results),'frames',flush=True)
if __name__=='__main__':
    for kind in ['all_people','before_after','prompts_predictions']:join(kind)
    for cid in ['04','07','10']:stages(cid)
    verify()
