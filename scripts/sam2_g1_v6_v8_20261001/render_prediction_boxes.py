"""Re-render existing frozen tracking outputs with pre-observation predictions.

No inference, state updates or new predictions. Geometry comes exclusively from
the saved prior.box and actual candidate.prompt_box at each displayed frame.
"""
from pathlib import Path
import sys,json,gzip,collections
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
DEST=OUT/'presentation/prediction_boxes'
sys.path.insert(0,str(ROOT/'scripts/sam2_negative_prompts_20261001'))
from pilot import decode
from common import read_json,write_json,digest,encoder,finish_encoder
BRANCH='motion_with_camera'
COLORS={'A':(70,240,80),'B':(255,180,50),'C':(240,80,230),'D':(70,210,250)}
STATUS={'INITIALIZED':'初始标注','TRACKED':'输出','OBSERVATION_UNCERTAIN':'输出（存疑）','PENDING':'待确认','RECOVERED':'重新确认','LOST':'未输出','UNMATCHED':'未匹配','OUT_OF_VIEW_PRED':'预测出画'}
FONTS={n:ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',n) for n in [20,22,24,26,28,30]}
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]
def finite_box(b):return b is not None and np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]
def center(b):return (np.asarray(b[:2],float)+np.asarray(b[2:],float))/2
def inside(p,w,h):return np.isfinite(p).all() and 0<=p[0]<w and 0<=p[1]<h
def point(p):return tuple(np.rint(p).astype(int))
def segment(im,a,b,col,thickness=1):
    a=np.clip(np.rint(a),-1000000,1000000).astype(int);b=np.clip(np.rint(b),-1000000,1000000).astype(int)
    ok,p,q=cv2.clipLine((0,0,im.shape[1],im.shape[0]),tuple(a),tuple(b))
    if ok:cv2.line(im,p,q,col,thickness,cv2.LINE_AA)
def dashed(im,box,col):
    if not finite_box(box):return
    x0,y0,x1,y1=np.asarray(box,float);h,w=im.shape[:2]
    # Only real box edges are shown. Never invent an edge by clamping an
    # off-image coordinate to the picture border.
    for y in [y0,y1]:
        if 0<=y<h:
            for x in range(max(0,int(np.floor(x0))),min(w,int(np.ceil(x1))),18):
                segment(im,[x,y],[min(x+10,x1),y],col,2)
    for x in [x0,x1]:
        if 0<=x<w:
            for y in range(max(0,int(np.floor(y0))),min(h,int(np.ceil(y1))),18):
                segment(im,[x,y],[x,min(y+10,y1)],col,2)
def corners(im,box,col):
    if not finite_box(box):return
    x0,y0,x1,y1=np.asarray(box,float);length=max(4,min(13,(x1-x0)/4,(y1-y0)/4))
    for x,sx in [(x0,1),(x1,-1)]:
        for y,sy in [(y0,1),(y1,-1)]:
            segment(im,[x,y],[x+sx*length,y],col,1);segment(im,[x,y],[x,y+sy*length],col,1)
def label(im,s,p,col):
    x,y=point(p);h,w=im.shape[:2];x=max(3,min(w-62,x));y=max(24,min(h-4,y))
    (tw,th),_=cv2.getTextSize(s,cv2.FONT_HERSHEY_SIMPLEX,.64,2)
    cv2.rectangle(im,(x-2,y-th-4),(min(w-1,x+tw+3),y+4),(12,12,12),-1)
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.64,col,2,cv2.LINE_AA)
def prompt_boxes(branch):
    return {c['owner']:c['prompt_box'] for c in branch['candidates'] if c['source']=='motion_box' and c['prompt_box'] is not None}
def layer(bgr,rows,branch,width):
    h=width*3//4;scale=width/2560;im=cv2.resize(bgr,(width,h),interpolation=cv2.INTER_AREA)
    masks={r['id']:cv2.resize(decode(r['rle']).astype(np.uint8),(width,h),interpolation=cv2.INTER_NEAREST) for r in rows}
    counts=sum(m.astype(np.uint8) for m in masks.values())
    for i,m in masks.items():
        region=(m>0)&(counts==1);im[region]=(.70*im[region]+.30*np.array(COLORS[i])).astype(np.uint8)
        contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,COLORS[i],2)
    yy,xx=np.indices((h,width));im[(counts>1)&((xx+yy)%12<3)]=230
    prompts=prompt_boxes(branch)
    for i,d in branch['decisions'].items():
        col=COLORS[i];prior=d.get('prior');box=prior.get('box') if prior else None
        p=None
        if finite_box(box):
            p=center(box)*scale;dashed(im,np.asarray(box)*scale,col)
            if inside(p,width,h):
                cv2.drawMarker(im,point(p),col,cv2.MARKER_TILTED_CROSS,15,2)
                # P distinguishes predicted geometry when an identity has no output mask.
                b=np.asarray(box)*scale
                label(im,i+' P',[b[0],b[1]-8] if inside([b[0],b[1]],width,h) else p+[8,-10],col)
        if i in prompts:corners(im,np.asarray(prompts[i])*scale,col)
        obs=d.get('observation_box')
        if finite_box(obs):
            q=center(obs)*scale
            if inside(q,width,h):
                if p is not None:segment(im,p,q,col,1)
                cv2.circle(im,point(q),5,(12,12,12),3,cv2.LINE_AA);cv2.circle(im,point(q),4,col,2,cv2.LINE_AA)
            if p is None:label(im,i,[obs[0]*scale,obs[1]*scale-8],col)
    return im
def paint(canvas,rows):
    im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
    for x,y,t,size,color in rows:d.text((x,y),t,font=FONTS[size],fill=color)
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
def dims(b):return '—' if not finite_box(b) else f'{b[2]-b[0]:.0f}×{b[3]-b[1]:.0f}'
def single(bgr,masks,branch,cid,f,meta):
    canvas=np.full((1120,1760,3),20,np.uint8);canvas[64:1024,:1280]=layer(bgr,masks,branch,1280)
    text=[(18,12,f'{cid}  |  第6版：一步预测位置与 SAM2 跟踪',30,'white'),
          (1300,16,f"原片 {meta['source']['start_s']+f/10:.1f} 秒",24,'white')]
    prompts=prompt_boxes(branch)
    for n,(i,d) in enumerate(branch['decisions'].items()):
        y=82+n*225;cv2.rectangle(canvas,(1294,y),(1744,y+211),(34,34,34),-1);col=tuple(reversed(COLORS[i]))
        prior=d.get('prior');pbox=prior.get('box') if prior else None;obs=d.get('observation_box')
        delta=float(np.linalg.norm(center(pbox)-center(obs))) if finite_box(pbox) and finite_box(obs) else None
        source=d.get('selected_candidate') or ''
        source='运动框提示' if source.endswith('motion_box') else '原生记忆' if source.endswith('memory') else '初始 mask' if source=='initial' else '无'
        score='—' if d.get('score') is None else f"{d['score']:.2f}"
        gap='—' if delta is None else f'{delta:.0f} px'
        text.extend([(1306,y+8,f"{i} {'主目标' if i=='A' else '邻近人物'}  ·  {STATUS[d['status']]}",24,col),
                     (1306,y+44,f'输出来源：{source}    评分 {score}',20,'white'),
                     (1306,y+76,f'预测—观测中心距离：{gap}',20,'white'),
                     (1306,y+108,f"框提示：{'启用' if i in prompts else '停用'}    记忆：{'写入' if d['memory_written'] else '保持'}",20,'white'),
                     (1306,y+140,f'预测框：{dims(pbox)} px',20,(210,210,210)),
                     (1306,y+172,f'实际提示框：{dims(prompts.get(i))} px',20,(210,210,210))])
    text.extend([(18,1035,'虚线框 / ×：一步预测位置    彩色 mask / ○：接收观测    连线：预测与观测中心差距',24,'white'),
                 (18,1075,'细角标：实际启用的 SAM2 框提示；预测框与提示框不同。首帧仅初始化；距离按原始 2560×1920 像素计算。',20,(195,195,195))])
    return paint(canvas,text)
def compare(bgr,masks,branches,cid,f,meta):
    canvas=np.full((1000,1920,3),20,np.uint8);text=[]
    for k,v in enumerate(['v6','v8']):
        branch=branches[v];canvas[64:784,k*960:(k+1)*960]=layer(bgr,masks[v],branch,960)
        text.append((18+k*960,12,f"{cid}  |  {'第6版 · 固定运动权重20%' if v=='v6' else '第8版 · 身份与运动可靠性'}",28,'white'))
        ds=branch['decisions'];status='   '.join(f"{i}: {STATUS[d['status']]}" for i,d in ds.items())
        prompts=' '.join(prompt_boxes(branch)) or '无';written=' '.join(i for i,d in ds.items() if d['memory_written']) or '无'
        text.extend([(18+k*960,796,status,22,'white'),(18+k*960,830,f'实际框提示：{prompts}     写入记忆：{written}',20,(210,210,210))])
    text.extend([(18,874,'虚线框 / ×：一步预测    彩色 mask / ○：接收观测    连线：中心差距    细角标：实际 SAM2 框提示',24,'white'),
                 (18,914,'两版均显示保存的一步预测框；即使第8版停用框提示，预测仍可存在。超出画面的部分保留原尺寸，仅画可见边。',20,(200,200,200)),
                 (18,954,f"原片 {meta['source']['start_s']+f/10:.1f} 秒  |  片段 {f/10:.1f} 秒  |  帧 {f}/{meta['frames']-1}  |  A 为主目标；无 mask 不代表人物消失。",22,'white')])
    return paint(canvas,text)
def run():
    DEST.mkdir(exist_ok=True);assert not (DEST/'manifest.json').exists(),'Preserve completed rendering.'
    all6,log6=encoder(DEST/'all_10_v6_predictions.mp4',1760,1120,10,19)
    both,logb=encoder(DEST/'all_10_v6_v8_predictions.mp4',1920,1000,10,19)
    manifest=[];sources={};count=0
    for k in range(1,11):
        cid=f'{k:02d}';folder=OUT/cid;meta=read_json(folder/'input/metadata.json');logs={};masks={}
        for v in ['v6','v8']:
            for name in ['decisions.jsonl.gz','masks_rle.jsonl.gz']:
                p=folder/v/name;sources[str(p.relative_to(ROOT))]=digest(p)
            logs[v]=load(folder/v/'decisions.jsonl.gz');masks[v]=collections.defaultdict(list)
            for r in load(folder/v/'masks_rle.jsonl.gz'):masks[v][r['frame']].append(r)
        cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'))
        one,ol=encoder(DEST/f'{cid}_v6_predictions.mp4',1760,1120,10,19)
        pair,pl=encoder(DEST/f'{cid}_v6_v8_predictions.mp4',1920,1000,10,19)
        for f in range(meta['frames']):
            ok,bgr=cap.read();assert ok
            branches={v:logs[v][f]['branches'][BRANCH] for v in logs}
            for v in branches:
                assert logs[v][f]['frame']==f
                for d in branches[v]['decisions'].values():
                    if d.get('prior'):assert d['prior']['last_observation_frame']<f
            frame6=single(bgr,masks['v6'][f],branches['v6'],cid,f,meta)
            frameb=compare(bgr,{v:masks[v][f] for v in masks},branches,cid,f,meta)
            one.stdin.write(frame6.tobytes());all6.stdin.write(frame6.tobytes())
            pair.stdin.write(frameb.tobytes());both.stdin.write(frameb.tobytes());count+=1
            if f in {meta['frames']//2,meta['frames']-1}:
                cv2.imwrite(str(DEST/f'{cid}_{f:03d}_v6.jpg'),frame6,[cv2.IMWRITE_JPEG_QUALITY,94])
                cv2.imwrite(str(DEST/f'{cid}_{f:03d}_comparison.jpg'),frameb,[cv2.IMWRITE_JPEG_QUALITY,94])
        cap.release();finish_encoder(one,ol);finish_encoder(pair,pl)
        manifest.append(dict(clip=cid,frames=meta['frames'],montage_start_seconds=(count-meta['frames'])/10))
        print('Rendered prediction boxes',cid,flush=True)
    finish_encoder(all6,log6);finish_encoder(both,logb);assert count==530
    for p,sha in sources.items():assert digest(ROOT/p)==sha
    write_json(DEST/'manifest.json',dict(frames=530,fps=10,duration_seconds=53,clips=manifest,sources_unchanged=True,source_hashes=sources,
        prediction='Saved prior.box and center, computed before the current RGB observation; current-frame offline Aria rotation projection.',
        prompt='Actual candidate.prompt_box, drawn as thin corner marks only when that candidate was generated.',
        observed='Final accepted mask; small circle denotes its bounding-box center, not mask centroid.',
        distance='Distance between prior.box center and accepted observation_box center in original 2560x1920 pixels; not GT error.',
        geometry='Full linear-camera frame, no crop; proportional resize only; off-image edges omitted without altering box dimensions.',
        inference='No model rerun, changes to masks, temporal interpolation, or state updates. Initial frames have no prediction.',
        renderer_sha256=digest(Path(__file__))))
if __name__=='__main__':run()
