"""Review masks, lifecycle, repairs, and actual V6 motion proposals."""
from pathlib import Path
import sys,importlib.util,collections,argparse
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_near_people_20261002';SRC=ROOT/'outputs/sam2_g1_v6_v8_20261001';OLD=ROOT/'outputs/sam2_incremental_v6_20261002'
spec=importlib.util.spec_from_file_location('old_render',ROOT/'scripts/sam2_incremental_v6_20261002/render.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
from common import read_json,write_json,encoder,finish_encoder

def render(cid,revision='raw',presentation=None):
    folder=OUT/cid;res=folder if revision=='raw' else folder/revision
    if not (res/'masks_rle.jsonl.gz').exists():return
    dest=OUT/'presentation'/(presentation or revision);dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'{cid}_media.json').exists():return
    census=read_json((res if (res/'census.json').exists() else folder)/'census.json');entries=census['segments'];ids=sorted({e['id'] for e in entries});cols={i:v.color(int(i[1:])-1) for i in ids}
    masks=collections.defaultdict(list)
    for r in v.load(res/'masks_rle.jsonl.gz'):masks[r['frame']].append(r)
    logs={(r['frame'],r['id']):r for r in v.load(res/'decisions.jsonl.gz')}
    oldpath=OLD/cid
    if cid=='10':oldpath=OLD/'reviewed'/cid
    old=collections.defaultdict(list)
    for r in v.load(oldpath/'masks_rle.jsonl.gz'):old[r['frame']].append(r)
    oldids=sorted({r['id'] for rr in old.values() for r in rr});oldcols={i:v.color(j) for j,i in enumerate(oldids)}
    cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));enc,elog=encoder(dest/f'{cid}_near_masks.mp4',1600,1060,10,20);cmp,clog=encoder(dest/f'{cid}_before_after.mp4',1920,820,10,20);diag,dlog=encoder(dest/f'{cid}_motion_prompts.mp4',1600,1060,10,20)
    samples=sorted(set([0,N//4,N//2,3*N//4,N-1]));cards=[];coverage=[]
    for f in range(N):
        ok,frame=cap.read();assert ok
        vis,visible=v.overlay(frame,masks[f],cols,logs,f);dv,_=v.overlay(frame,masks[f],cols,logs,f,True);before,_=v.overlay(frame,old[f],oldcols,{},f)
        expected=[e['id'] for e in entries if e['start']<=f<=e['end']];canvas=np.zeros((1060,1600,3),np.uint8);canvas[70:1030,:1280]=vis
        lines=[(18,14,f'{cid} | 近处优先 · 连续进出标注 | {f/10:.1f} 秒'),(1300,85,f'登记 {len(ids)} 人 / 无总数上限'),(1300,120,f'本帧 {len(visible)} 个 mask')]
        for j,i in enumerate(ids):
            d=logs.get((f,i),{});s=d.get('status','');state='标注中' if i in visible else ('遮挡/待核查' if i in expected else '未标注区间')
            if s=='CENSUS_PROMPT_REPAIR':state='修正后标注'
            if s.startswith('REVIEW') and i in visible:state='回查后标注'
            lines.append((1300,195+j*47,f'{i}  {state}'))
        lines.append((18,1033,'离线双向标注；仅标注当前可见区域。远处暂不覆盖；彩色 mask 仍需人工验收。'))
        canvas=v.text(canvas,lines,20);enc.stdin.write(canvas.tobytes());dc=canvas.copy();dc[70:1030,:1280]=dv;dc[1030:]=0;dc=v.text(dc,[(18,1033,'虚线：当前处理方向的 V6 预测位置；红叉：实际负点。倒放处理方向使用后续原始帧。')],20);diag.stdin.write(dc.tobytes())
        comp=np.zeros((820,1920,3),np.uint8);comp[60:780,:960]=cv2.resize(before,(960,720));comp[60:780,960:]=cv2.resize(vis,(960,720));comp=v.text(comp,[(20,10,f'{cid} 上次：不限制人数的增量标注'),(980,10,'本次：近处优先 + 人物区间核查 + 双向补标'),(20,787,'同一视频、同一时刻；右侧改变了标注范围与核查流程，不能据此单独归因于某个模块。')],22);cmp.stdin.write(comp.tobytes())
        if f in samples:
            cv2.imwrite(str(dest/f'{cid}_{f:03d}.jpg'),canvas);cards.append(cv2.resize(canvas,(800,530)))
        coverage.append(dict(frame=f,expected=expected,visible=visible))
    cap.release()
    for p,l in [(enc,elog),(cmp,clog),(diag,dlog)]:finish_encoder(p,l)
    sheet=np.zeros((1590,1600,3),np.uint8)
    for j,im in enumerate(cards):sheet[j//2*530:(j//2+1)*530,j%2*800:(j%2+1)*800]=im
    cv2.imwrite(str(dest/f'{cid}_contact.jpg'),sheet);write_json(dest/f'{cid}_media.json',dict(frames=N,coverage=coverage));print(cid,revision,'rendered',flush=True)
    # Every-five-frame person crops, including pending gaps, expose identities clearly.
    cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        frames.append(im)
    cap.release()
    for e in entries:
        samples=sorted(set([e['start'],e['end'],e['anchor_frame']]+list(range(e['start'],e['end']+1,5))))
        from PIL import Image,ImageDraw,ImageFont
        sheet=Image.new('RGB',(8*220,((len(samples)+7)//8)*250),(240,240,240));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(v.FONT,16)
        for j,f in enumerate(samples):
            r=next((r for r in masks[f] if r['id']==e['id']),None);guide=next(r['box'] for r in e['rows'] if r['frame']==f);b=np.array(guide).astype(int);pad=max(15,int(.12*(b[3]-b[1])));x1,y1=max(0,b[0]-pad),max(0,b[1]-pad);x2,y2=min(2560,b[2]+pad),min(1920,b[3]+pad)
            im=frames[f][y1:y2,x1:x2].copy()
            if r:
                m=v.decode(r['rle'])[y1:y2,x1:x2].astype(bool);im[m]=(.65*im[m]+.35*np.array(cols[e['id']])).astype(np.uint8)
                cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,cols[e['id']],2)
            crop=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));crop.thumbnail((215,215));x=j%8*220;y=j//8*250;sheet.paste(crop,(x,y));draw.text((x,y+217),f"f{f} {'MASK' if r else 'PENDING'}",fill=(0,0,0),font=font)
        sheet.save(dest/f'{cid}_{e["segment"]}_strip.jpg',quality=94)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=[f'{i:02d}' for i in range(1,11)]);p.add_argument('--revision',default='raw');p.add_argument('--presentation');a=p.parse_args()
    for cid in a.clips:render(cid,a.revision,a.presentation)
