import render as renderer
from render import *
from common import ffmpeg,winpath,digest
import subprocess
def metrics(rows):
    grouped=collections.defaultdict(list);ids=collections.Counter();conflicts=[]
    for r in rows:grouped[r['frame']].append(r)
    for f,rs in grouped.items():
        decoded=[]
        for r in rs:
            m=cv2.resize(decode(r['rle']),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():ids[r['id']]+=1;decoded.append((r['id'],m,int(m.sum())))
        for j,(i,m,n) in enumerate(decoded):
            for other,mask,nn in decoded[:j]:
                inter=np.count_nonzero(m&mask)
                if inter/max(1,min(n,nn))>.65 and min(n,nn)/max(n,nn)>.15:conflicts.append(dict(frame=f,ids=[i,other]))
    return dict(nonempty=sum(ids.values()),per_id=dict(ids),overlap_pair_frames=len(conflicts),conflicts=conflicts)
def control():
    dest=OUT/'controls';primary=load(OUT/'04/masks_rle.jsonl.gz');off=load(dest/'no_negative/04/masks_rle.jsonl.gz');a={(r['frame'],r['id']):r['rle'] for r in primary};b={(r['frame'],r['id']):r['rle'] for r in off};assert a.keys()==b.keys()
    results=dict(negative_on=metrics(primary),negative_off=metrics(off),identical_rows=sum(a[k]==b[k] for k in a),rows=len(a),selection='Same seeds and batch plan from 04, chosen for historical merge failure. No GT; overlap is not accuracy.')
    write_json(dest/'comparison.json',results);print(results,flush=True)
    inv=read_json(OUT/'04/inventory.json');cols={e['id']:color(j) for j,e in enumerate(inv['entries'])};grouped=[]
    for rows in [off,primary]:
        g=collections.defaultdict(list)
        for r in rows:g[r['frame']].append(r)
        grouped.append(g)
    target=dest/'04_negative_on_off.mp4'
    if not target.exists():
        cap=cv2.VideoCapture(str(SRC/'04/input/rgb.mp4'));enc,log=encoder(target,1920,820,10,20)
        for f in range(40):
            ok,frame=cap.read();assert ok;canvas=np.zeros((820,1920,3),np.uint8)
            for j,g in enumerate(grouped):vis,_=overlay(frame,g[f],cols,{},f);canvas[60:780,j*960:(j+1)*960]=cv2.resize(vis,(960,720))
            canvas=text(canvas,[(20,10,'04 同种子、同批次：关闭负点'),(980,10,'04 同种子、同批次：开启负点'),(20,786,'仅关闭/开启跨批负点；双方沿用第6版评分和记忆规则。自动重叠计数不等于身份准确率。')],23);enc.stdin.write(canvas.tobytes())
            if f in [0,10,20,30,39]:cv2.imwrite(str(dest/f'comparison_{f:03d}.jpg'),canvas)
        cap.release();finish_encoder(enc,log)
def refined():
    mainout=OUT;renderer.OUT=OUT/'reviewed';renderer.render('10')
    for kind in ['all_people','before_after','prompts_predictions']:
        target=mainout/'presentation'/f'all_10_{kind}_reviewed.mp4'
        if target.exists():continue
        paths=[mainout/'presentation'/f'{i:02d}_{kind}.mp4' for i in range(1,10)]+[renderer.OUT/'presentation'/f'10_{kind}.mp4']
        listing=target.with_suffix('.concat.txt');listing.write_text('\n'.join("file '"+winpath(p).replace('\\','/')+"'" for p in paths),encoding='utf8')
        subprocess.run([str(ffmpeg()),'-hide_banner','-loglevel','error','-nostdin','-f','concat','-safe','0','-i',winpath(listing),'-c','copy','-movflags','+faststart','-n',winpath(target)],check=True)
def verify():
    paths=list((OUT/'presentation').glob('*_reviewed.mp4'))+list((OUT/'reviewed/presentation').glob('*.mp4'))+[OUT/'controls/04_negative_on_off.mp4'];results=[]
    for p in paths:
        cap=cv2.VideoCapture(str(p));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            n+=1
        cap.release();expected=530 if p.name.startswith('all_10') else (40 if p.name.startswith('04') else 50);assert n==expected;results.append(dict(path=str(p.relative_to(OUT)),frames=n,sha256=digest(p)))
    write_json(OUT/'additional_media_verification.json',dict(status='passed',videos=results));print('extras verified',len(results),flush=True)
if __name__=='__main__':control();refined();verify()
