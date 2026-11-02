"""Complete-clip incremental masks and actual prompt/motion diagnostics."""
import sys,json,gzip,collections,colorsys,argparse
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_incremental_v6_20261002';SRC=ROOT/'outputs/sam2_g1_v6_v8_20261001'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,encoder,finish_encoder
W,H=2560,1920
FONT='C:/Windows/Fonts/msyh.ttc' if sys.platform=='win32' else '/mnt/c/Windows/Fonts/msyh.ttc'
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def decode(rle):
    m=np.zeros((H,W),np.uint8)
    if not rle:return m
    counts=rle[:-4];x1,y1,x2,y2=map(int,rle[-4:]);bw=x2-x1+1;bh=y2-y1+1
    if bw<=0 or bh<=0:return m
    flat=np.repeat(np.arange(len(counts))%2,np.array(counts,dtype=np.int64)).astype(np.uint8)
    m[y1:y2+1,x1:x2+1]=flat.reshape(bh,bw);return m
def color(index):
    if index<4:return [(70,240,80),(255,180,50),(240,80,230),(70,210,250)][index]
    r,g,b=colorsys.hsv_to_rgb((index*.618)%1,.72,1);return (int(b*255),int(g*255),int(r*255))
def text(im,lines,size=24):
    p=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p);font=ImageFont.truetype(FONT,size)
    for x,y,s in lines:d.text((x,y),s,fill='white',font=font)
    return cv2.cvtColor(np.array(p),cv2.COLOR_RGB2BGR)
def dash(im,box,c,scale=.5):
    if box is None:return
    b=np.array(box)*scale
    if not np.isfinite(b).all() or np.max(np.abs(b))>100000:return
    x1,y1,x2,y2=map(int,b)
    for a,b in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
        a=np.array(a,float);b=np.array(b,float);length=np.linalg.norm(b-a)
        if length<1:continue
        for t in range(0,int(length),15):
            p=a+(b-a)*t/length;q=a+(b-a)*min(t+8,length)/length
            cv2.line(im,tuple(p.astype(int)),tuple(q.astype(int)),c,1,cv2.LINE_AA)
def overlay(frame,rows,cols,logs,f,diagnostic=False):
    im=cv2.resize(frame,(1280,960));unions={};visible=[]
    for row in rows:
        mask=cv2.resize(decode(row['rle']),(1280,960),interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():continue
        identity=row['id'];c=cols[identity];im[mask]=(.7*im[mask]+.3*np.array(c)).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,c,1)
        yy,xx=np.where(mask);cv2.putText(im,identity,(int(xx.min()),max(12,int(yy.min())-4)),0,.47,c,1,cv2.LINE_AA);visible.append(identity)
        if diagnostic:
            d=logs.get((f,identity),{});prior=d.get('prior')
            if prior:dash(im,prior.get('box'),c)
            for p in d.get('negative_points',[]):cv2.drawMarker(im,tuple((np.array(p['xy'])*.5).astype(int)),(20,30,255),cv2.MARKER_TILTED_CROSS,8,2)
    return im,visible
def render(cid):
    folder=OUT/cid;inv=read_json(folder/'inventory.json');dest=OUT/'presentation';dest.mkdir(exist_ok=True)
    if (dest/f'{cid}_media.json').exists():return
    masks=collections.defaultdict(list)
    for r in load(folder/'masks_rle.jsonl.gz'):masks[r['frame']].append(r)
    decisions=load(folder/'decisions.jsonl.gz');logs={(r['frame'],r['id']):r for r in decisions};cols={e['id']:color(j) for j,e in enumerate(inv['entries'])}
    old=collections.defaultdict(list)
    for r in load(SRC/cid/'v6/masks_rle.jsonl.gz'):old[r['frame']].append(r)
    oldcols={i:color(j) for j,i in enumerate(read_json(SRC/cid/'input/initial_masks.json'))}
    cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));samples=[0,N//4,N//2,3*N//4,N-1]
    enc,elog=encoder(dest/f'{cid}_all_people.mp4',1600,1060,10,20)
    comp,clog=encoder(dest/f'{cid}_before_after.mp4',1920,820,10,20)
    boxes,blog=encoder(dest/f'{cid}_prompts_predictions.mp4',1600,1060,10,20)
    cards=[];stats=dict(frames=N,provisional_ids=len(inv['entries']),batches=inv['batches'],visible_per_frame=[],negative_points_selected=0)
    for f in range(N):
        ok,frame=cap.read();assert ok
        vis,ids=overlay(frame,masks[f],cols,logs,f);diag,_=overlay(frame,masks[f],cols,logs,f,True);left,_=overlay(frame,old[f],oldcols,{},f)
        canvas=np.zeros((1060,1600,3),np.uint8);canvas[70:1030,:1280]=vis
        lines=[(18,14,f'{cid} | 第6版 · 增量多人标注 | {f/10:.1f} 秒'),(1300,82,f'累计 {len(inv["entries"])} 个候选 ID'),(1300,120,f'分 {inv["batches"]} 批完成'),(1300,158,f'本帧 {len(ids)} 个 mask')]
        for j,e in enumerate(inv['entries']):
            x=1300+(j//26)*145;y=215+(j%26)*29
            if x>1550:break
            marker='●' if e['id'] in ids else '·';lines.append((x,y,f'{marker} {e["id"]}'))
        lines.extend([(18,1032,'颜色与编号表示候选身份；自动标注仍可能漏跟、重复或串身份。')]);canvas=text(canvas,lines,22);enc.stdin.write(canvas.tobytes())
        dia=canvas.copy();dia[70:1030,:1280]=diag;dia=text(dia,[(18,1032,'虚线：当前帧预测位置；红叉：实际用于所选候选的前批负点。')],22);boxes.stdin.write(dia.tobytes())
        comparison=np.zeros((820,1920,3),np.uint8);comparison[60:780,:960]=cv2.resize(left,(960,720));comparison[60:780,960:]=cv2.resize(vis,(960,720))
        comparison=text(comparison,[(20,10,f'{cid} 原第6版：{len(oldcols)} 个初始目标'),(980,10,f'增量第6版：{len(inv["entries"])} 个候选 ID，{inv["batches"]} 批'),(20,787,'两侧同一原图、同一时刻；新增目标与负点同时变化，本视频不单独证明负点的因果效果。')],23);comp.stdin.write(comparison.tobytes())
        if f in samples:
            cv2.imwrite(str(dest/f'{cid}_{f:03d}_all.jpg'),canvas);cv2.imwrite(str(dest/f'{cid}_{f:03d}_prompts.jpg'),dia);cards.append(cv2.resize(canvas,(800,530)))
        stats['visible_per_frame'].append(len(ids));stats['negative_points_selected']+=sum(len(logs.get((f,i),{}).get('negative_points',[])) for i in cols)
    cap.release()
    for p,l in [(enc,elog),(comp,clog),(boxes,blog)]:finish_encoder(p,l)
    contact=np.zeros((530*3,1600,3),np.uint8)
    for j,card in enumerate(cards):contact[j//2*530:(j//2+1)*530,j%2*800:(j%2+1)*800]=card
    cv2.imwrite(str(dest/f'{cid}_contact.jpg'),contact);write_json(dest/f'{cid}_media.json',stats);print(cid,stats,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=[f'{i:02d}' for i in range(1,11)]);args=p.parse_args()
    for c in args.clips:
        if (OUT/c/'inventory.json').exists():render(c)
