"""Synchronized masks, same spatial crop, actual status; no fabricated GT."""
import sys,json,gzip,collections
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
sys.path.insert(0,str(ROOT/'scripts/sam2_negative_prompts_20261001'))
from pilot import decode
from common import read_json,write_json,encoder,finish_encoder
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
COLORS=[(70,240,80),(255,180,50),(240,80,230),(70,210,250)]
BRANCH='motion_with_camera'
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]
def title(bgr,lines,size=25):
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,size)
    for x,y,t in lines:d.text((x,y),t,fill='white',font=font)
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
def overlay(bgr,rows):
    vis=bgr.copy()
    for row in rows:
        mask=decode(row['rle']);j=ord(row['id'])-65;color=COLORS[j]
        if not mask.any():continue
        vis[mask]=(.65*vis[mask]+.35*np.array(color)).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,color,3)
        yy,xx=np.where(mask);cv2.putText(vis,row['id'],(int(xx.min()),max(30,int(yy.min())-7)),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3)
    return vis
def render_clip(cid):
    folder=OUT/cid;dest=folder/'comparison';dest.mkdir(exist_ok=True)
    if (dest/'stats.json').exists():return
    meta=read_json(folder/'input/metadata.json');N=meta['frames'];initial=read_json(folder/'input/initial_masks.json')
    logs={v:load(folder/v/'decisions.jsonl.gz') for v in ['v6','v8']}
    masks={v:collections.defaultdict(list) for v in logs}
    stats={v:dict(ids={i:dict(nonempty=0,post_seed_nonempty=0,memory_writes=0,prompts=0,statuses={},empty_frames=[]) for i in initial}) for v in logs}
    boxes=[]
    for v in logs:
        assert len(logs[v])==N
        for row in load(folder/v/'masks_rle.jsonl.gz'):masks[v][row['frame']].append(row)
        for f,row in enumerate(logs[v]):
            assert row['frame']==f and row['current_camera_pose_frame']==f
            assert len(masks[v][f])==len(initial)
            for i,d in row['branches'][BRANCH]['decisions'].items():
                s=stats[v]['ids'][i];s['statuses'][d['status']]=s['statuses'].get(d['status'],0)+1
                present=d['observation_box'] is not None;s['nonempty']+=int(present)
                if f:
                    s['post_seed_nonempty']+=int(present);s['memory_writes']+=int(d['memory_written'])
                    s['prompts']+=int(any(c['owner']==i and c['source']=='motion_box' for c in row['branches'][BRANCH]['candidates']))
                if not present:s['empty_frames'].append(f)
                else:boxes.append(d['observation_box'])
                assert max(d['memory_after']['references']+d['memory_after']['recent'])<=f
        assert masks[v][0]==masks['v6'][0]
        p=read_json(folder/v/'protocol.json')
        for key in ['rgb','initial_masks','pose','alignment']:
            assert p['source_hashes'][key]['sha256']==read_json(folder/'v6/protocol.json')['source_hashes'][key]['sha256']
    b=np.array(boxes);x1=max(0,int(b[:,0].min())-90);x2=min(2560,int(b[:,2].max())+90)
    y1=max(0,int(b[:,1].min())-90);y2=min(1920,int(b[:,3].max())+90)
    # Add letterboxing to retain aspect ratio; identical crop for both methods and every frame.
    def panel(im):
        im=im[y1:y2,x1:x2];h,w=im.shape[:2];s=min(960/w,720/h)
        resized=cv2.resize(im,(int(w*s),int(h*s)));out=np.zeros((720,960,3),np.uint8)
        oy=(720-resized.shape[0])//2;ox=(960-resized.shape[1])//2;out[oy:oy+resized.shape[0],ox:ox+resized.shape[1]]=resized
        return out
    cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'));enc,elog=encoder(dest/'comparison.mp4',1920,880,10,20)
    samples=sorted(set([0,N//4,N//2,3*N//4,N-1]));tiles=[]
    for f in range(N):
        ok,bgr=cap.read();assert ok
        vis=[overlay(bgr,masks[v][f]) for v in logs]
        canvas=np.zeros((880,1920,3),np.uint8);canvas[70:790,:960]=panel(vis[0]);canvas[70:790,960:]=panel(vis[1])
        lines=[(20,12,f'{cid} | 第6版 · 固定运动权重20%'),(980,12,f'{cid} | 第8版 · 身份与运动可靠性'),
               (20,840,f"原视频 {meta['source']['start_s']+f/10:.1f}s | 片段 {f/10:.1f}s | frame {f}/{N-1} | A 为原筛选目标；空缺不代表人物消失")]
        for k,v in enumerate(logs):
            ds=logs[v][f]['branches'][BRANCH]['decisions']
            status='   '.join(f"{i}: {d['status']}" for i,d in ds.items())
            lines.append((20+k*960,800,status))
        canvas=title(canvas,lines,23);enc.stdin.write(canvas.tobytes())
        if f in samples:
            cv2.imwrite(str(dest/f'frame_{f:03d}.jpg'),canvas)
            tiles.append(cv2.resize(canvas,(1280,587)))
            full=np.zeros((820,1920,3),np.uint8)
            full[60:780,:960]=cv2.resize(vis[0],(960,720));full[60:780,960:]=cv2.resize(vis[1],(960,720))
            full=title(full,[(12,10,f'{cid} frame {f}: V6'),(972,10,'V8')])
            cv2.imwrite(str(dest/f'full_{f:03d}.jpg'),full)
    cap.release();finish_encoder(enc,elog)
    cv2.imwrite(str(dest/'contact.jpg'),np.vstack(tiles))
    write_json(dest/'stats.json',dict(candidate=int(cid),frames=N,people=len(initial),crop_xyxy=[x1,y1,x2,y2],sample_frames=samples,branches=stats,
        note='Nonempty output and algorithm status are diagnostics, not GT correctness. Review initial masks and sampled original image context separately.'))
    print('Rendered',cid,flush=True)
def montage():
    dest=OUT/'comparison_all_10.mp4'
    if dest.exists():return
    enc,log=encoder(dest,1920,880,10,20);count=0
    for k in range(1,11):
        cap=cv2.VideoCapture(str(OUT/f'{k:02d}'/'comparison/comparison.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            enc.stdin.write(frame.tobytes());count+=1
        cap.release()
    finish_encoder(enc,log);assert count==530
    print('Combined',count,'frames',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:
        for cid in sys.argv[1:]:render_clip(cid)
    else:
        for k in range(1,11):render_clip(f'{k:02d}')
        montage()
