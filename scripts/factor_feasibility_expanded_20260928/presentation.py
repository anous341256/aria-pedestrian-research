"""Presentation-only rendering of frozen examples; no new forecasts or selection."""
import sys, gzip, json, math, subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare import ROOT, SRC, OUT, read, write, sha
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder, finish_encoder, ffmpeg, winpath

DEST=OUT/'presentation'
W,H=1600,1000
PW,PH=768,576
SCALE=PW/2560
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
FONTS={n:ImageFont.truetype(FONT,n) for n in (19,21,23,25,28,32,38,46)}
INK=(25,36,52); MUTED=(88,102,118); PAPER=(248,250,252)
RAW=(222,111,40); COMP=(0,138,163); REF=(73,240,135)
SCENES={'151153':'站台 / 站厅','162538':'食品卖场','164557':'街道 / 商场','193709':'傍晚过街','201129':'夜间站外'}
NOTES={1:'贴边目标，适合看截断影响',2:'站厅转向，画面有运动模糊',3:'白天站台，转头影响直观',4:'食品卖场，保留的未改善窗口',5:'白天街道，人物大且清楚',6:'过街转向，贴边及视角变化',7:'商场走廊，人物和场景清楚',8:'傍晚跟随，目标接近画面边缘',9:'傍晚近人，存在运动模糊',10:'傍晚过街，旋转影响很明显',11:'傍晚远人，目标较小',12:'傍晚过街，目标较小',13:'夜间站外，明显模糊与截断',14:'夜间站外，补偿差异直观'}
SHORTLIST=[3,5,7,10,14,4]

def txt(d,xy,text,size=23,color=INK):
    d.text(xy,text,font=FONTS[size],fill=color)

def decode_small(rle):
    runs=np.asarray(rle[:-4],dtype=np.int64);x1,y1,x2,y2=rle[-4:]
    crop=np.repeat(np.arange(len(runs),dtype=np.uint8)%2,runs).reshape(y2-y1+1,x2-x1+1)
    full=np.zeros((1920,2560),np.uint8);full[y1:y2+1,x1:x2+1]=crop
    assert crop.sum()==runs[1::2].sum()
    return cv2.resize(full,(PW,PH),interpolation=cv2.INTER_NEAREST).astype(bool)

def mask_cache(rows):
    need={}
    for r in rows:
        need.setdefault(r['recording'],set()).update((r['track'],f) for f in r['history_frames']+r['future_frames'])
    result={};sources={}
    for sid, keys in need.items():
        p=SRC/sid/'sam2_raw_v1/masks_rle.jsonl.gz';sources[sid]=sha(p)
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                z=json.loads(line)
                if (z['id'],z['frame']) in keys:result[(sid,z['id'],z['frame'])]=decode_small(z['rle'])
        assert all((sid,tid,frame) in result for tid,frame in keys)
    return result,sources

def video_frames(row):
    first=row['history_frames'][0];last=row['future_frames'][-1]
    cap=cv2.VideoCapture(str(SRC/row['recording']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    ims={}
    for f in range(first,last+1):
        ok,im=cap.read();assert ok
        ims[f]=cv2.cvtColor(cv2.resize(im,(PW,PH),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    cap.release();return ims

def mark_panel(rgb,mask,target,pred,color):
    im=rgb.copy();im[mask]=(im[mask]*.84+np.asarray(REF)*.16).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im,contours,-1,REF,2,cv2.LINE_AA)
    actual=np.array(target)*SCALE
    a=tuple(np.rint(actual).astype(int));cv2.drawMarker(im,a,(10,30,20),cv2.MARKER_TILTED_CROSS,21,5,cv2.LINE_AA)
    cv2.drawMarker(im,a,REF,cv2.MARKER_TILTED_CROSS,19,3,cv2.LINE_AA)
    outside=False
    if pred is not None:
        p=np.asarray(pred,float)*SCALE;inside=0<=p[0]<PW and 0<=p[1]<PH
        q=np.rint(np.clip(p,[15,15],[PW-16,PH-16])).astype(int)
        # A clipped indicator is explicitly marked as offscreen; metrics retain original coordinates.
        cv2.line(im,a,tuple(q),(250,250,250),4,cv2.LINE_AA)
        cv2.line(im,a,tuple(q),color,2,cv2.LINE_AA)
        if inside:
            if color==RAW:
                cv2.circle(im,tuple(q),12,(250,250,250),5,cv2.LINE_AA);cv2.circle(im,tuple(q),12,color,3,cv2.LINE_AA)
            else:
                points=np.array([q+[-13,0],q+[0,-13],q+[13,0],q+[0,13]],np.int32)
                cv2.polylines(im,[points],True,(250,250,250),5,cv2.LINE_AA);cv2.polylines(im,[points],True,color,3,cv2.LINE_AA)
        else:
            direction=p-q;direction/=max(np.linalg.norm(direction),1e-8)
            normal=np.array([-direction[1],direction[0]])
            tri=np.array([q+direction*12,q-direction*10+normal*10,q-direction*10-normal*10],np.int32)
            cv2.fillPoly(im,[tri],color,cv2.LINE_AA);cv2.polylines(im,[tri],True,(255,255,255),2,cv2.LINE_AA)
            outside=True
    return im,outside

def frame_canvas(spec,row,idx,frames,masks):
    fs=row['history_frames']+row['future_frames'];f=fs[idx];future=idx>=6
    target=np.asarray(row['target_xy'][idx-6] if future else row['history_xy'][idx])
    sheet=Image.new('RGB',(W,H),PAPER);d=ImageDraw.Draw(sheet)
    n=spec['case_number'];txt(d,(24,13),f'{n:02d}  {SCENES[row["recording"]]}  ·  {row["key"]}',32)
    txt(d,(24,58),'同一匀速预测器  |  历史 0.5 秒 → 未来 1.5 秒  |  0.5 倍速回放',23,MUTED)
    txt(d,(24,96),'原始轨迹预测  ○',28,RAW)
    txt(d,(808,96),'旋转补偿预测  ◇  ·  Aria 参考旋转',28,COMP)
    errors=[]
    for x,m,col in [(24,'raw_cv',RAW),(808,'comp_cv_refcam',COMP)]:
        pred=np.asarray(row['predictions'][m][idx-6],float) if future else None
        im,off=mark_panel(frames[f],masks[(row['recording'],row['track'],f)],target,pred,col)
        sheet.paste(Image.fromarray(im),(x,140))
        if future:
            error=float(np.linalg.norm(pred-target));errors.append(error)
            text=f'当前中心误差  {error:.0f} px'+('  ·  预测点在画外' if off else '')
        else:text='正在观察历史：尚未预测';errors.append(None)
        txt(d,(x,735),text,25,col)
        txt(d,(x,779),f'本窗口平均误差 ADE  {row["metrics"][m]["ADE_1.5s"]:.1f} px',28)
    rel=(f-row['frame'])/10
    phase=f'未来 +{rel:.1f} 秒' if future else f'历史 {rel:.1f} 秒' if rel<0 else '预测起点 t=0'
    txt(d,(24,835),phase,25)
    txt(d,(1180,835),f'视频时间 {f/10:.1f}s  ·  帧 {f}',21,MUTED)
    x0,x1,y=260,1140,850
    d.line((x0,y,x1,y),fill=(206,214,223),width=6)
    boundary=x0+(x1-x0)*.25
    d.line((x0,y,boundary,y),fill=(72,137,192),width=6)
    xp=x0+(x1-x0)*idx/20;d.ellipse((xp-8,y-8,xp+8,y+8),fill=INK)
    txt(d,(24,889),'绿色轮廓：目标 SAM mask    ×：参考中心    ○ / ◇：预测中心    连线：当前误差',23)
    txt(d,(24,935),'Aria/MPS 参考旋转实验；未来画面仅用于核对。全部数值来自上一轮固定窗口，单位为原图像素。',21,MUTED)
    return cv2.cvtColor(np.asarray(sheet),cv2.COLOR_RGB2BGR)

def concat(paths,dest):
    listing=dest.with_suffix('.concat.txt')
    listing.write_text(''.join("file '"+winpath(p).replace('\\','/')+"'\n" for p in paths),encoding='utf-8')
    cmd=[str(ffmpeg()),'-hide_banner','-loglevel','error','-nostdin','-f','concat','-safe','0','-i',winpath(listing),'-c','copy','-movflags','+faststart','-n',winpath(dest)]
    subprocess.run(cmd,check=True)

def contact(entries,path):
    tw,th=800,553;canvas=Image.new('RGB',(tw*2,th*math.ceil(len(entries)/2)),PAPER);d=ImageDraw.Draw(canvas)
    for i,z in enumerate(entries):
        x=(i%2)*tw;y=(i//2)*th
        im=Image.open(DEST/z['poster']).resize((800,500),Image.Resampling.LANCZOS);canvas.paste(im,(x,y))
        txt(d,(x+14,y+505),NOTES[z['case_number']],23)
    canvas.save(path,quality=94)

def index(manifest):
    s=DEST/'START_HERE.md'
    lines=['# 轨迹预测动态展示候选','',
      '全部视频使用上一轮已经固定的 14 个展示窗口，未重新挑选起点或改变预测结果。每例 8 秒：观察历史、固定预测起点、半速核对未来、终点停留。左右是同一帧、同一目标。',
      '', '**展示条件：Aria/MPS 参考相机旋转。** 绿色轮廓是原始 SAM mask，绿色 × 是其外接框中心；橙色圆是原始轨迹预测，蓝色菱形是补偿轨迹预测。连线连接当前预测与当前参考点，未把不同相机时刻的坐标拼成空间路线。画外预测以边缘三角形提示，误差计算不裁剪。',
      '', '[先看 6 个展示候选（48 秒）](shortlist_6.mp4) · [看全部 14 例（112 秒）](all_14_examples.mp4) · [6 例选片总览](shortlist_contact.jpg)',
      '', '建议先看 **03、05、07**：分别是站台、白天街道、商场走廊，人物和场景较容易辨认。**10** 适合说明转头造成的画面运动，**14** 展示夜间情形。**04** 是保留的未改善窗口，可以作为失败案例。此推荐按展示清晰度和解释用途给出，不是重新抽取统计样本。',
      '', '每例都有独立视频、可放进汇报的终点对比图，以及上一轮的完整曲线图。下表 ADE 是该展示窗口的误差，不是该轨迹所有窗口的平均值。',
      '', '| 编号 | 场景与用途 | 视频 / 图片 | 原始 ADE | 补偿 ADE |', '|---|---|---|---:|---:|']
    for z in manifest:
        lines.append(f"| {z['case_number']:02d} · {z['key']} | {NOTES[z['case_number']]} | [播放]({z['video']}) / [对比图]({z['poster']}) / [曲线图](../examples/{z['case_number']:02d}_{z['key']}.png) | {z['raw_ADE']:.1f} | {z['comp_ADE']:.1f} |")
    lines+=['','[原完整实验报告](../../../reports/factor_feasibility_expanded_20260928/RESULTS.md)','',
      '合集中每个案例恰好占 8 秒。全部版按编号 01–14 排列；候选版依次为 03、05、07、10、14、04。独立 MP4 使用 H.264，可直接插入演示文稿。',
      '', '所有原始标注、图像与统计文件均保留。本次只读取既有预测，并解码对应 SAM mask 来制作演示素材。']
    s.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    DEST.mkdir(exist_ok=True);(DEST/'clips').mkdir(exist_ok=True);(DEST/'posters').mkdir(exist_ok=True)
    rows=read(OUT/'examples.json');specs={s['key']:s for s in read(OUT/'candidates.json')}
    input_hash={name:sha(OUT/name) for name in ['examples.json','summary.json','predictions.json','protocol.json']}
    masks,mask_hash=mask_cache(rows);manifest=[]
    print('MASKS',len(masks),flush=True)
    # 1 sec lead-in, 0.5x playback (4.2 sec), 1 sec cutoff pause, 1.8 sec terminal pause = 8 sec.
    order=[0]*10
    for i in range(21):
        order.extend([i]*2)
        if i==5:order.extend([5]*10)
    order.extend([20]*18);assert len(order)==80
    for row in rows:
        spec=specs[row['key']];stem=f"{spec['case_number']:02d}_{row['key']}"
        path=DEST/'clips'/f'{stem}.mp4';frames=video_frames(row)
        rendered=[frame_canvas(spec,row,i,frames,masks) for i in range(21)]
        cv2.imwrite(str(DEST/'posters'/f'{stem}.jpg'),rendered[-1],[cv2.IMWRITE_JPEG_QUALITY,95])
        # Fixed QA frames: start / cutoff / early future / endpoint.
        if spec['case_number'] in [3,4,5,14]:
            for i in [0,5,10]:cv2.imwrite(str(DEST/'posters'/f'{stem}_qa_{i:02d}.jpg'),rendered[i],[cv2.IMWRITE_JPEG_QUALITY,90])
        proc,log=encoder(path,W,H,10,19)
        for i in order:proc.stdin.write(rendered[i].tobytes())
        finish_encoder(proc,log)
        manifest.append({'case_number':spec['case_number'],'key':row['key'],'recording':row['recording'],'cutoff':row['frame'],
            'frame_range':[row['history_frames'][0],row['future_frames'][-1]],'video':str(path.relative_to(DEST)),
            'poster':f'posters/{stem}.jpg','duration_s':8,'fps':10,'frames':80,'raw_ADE':row['metrics']['raw_cv']['ADE_1.5s'],
            'comp_ADE':row['metrics']['comp_cv_refcam']['ADE_1.5s'],'display_condition':'Aria/MPS reference historical and future rotation',
            'video_sha256':sha(path)})
        print('RENDERED',stem,flush=True)
    write(DEST/'manifest.json',manifest)
    concat([DEST/z['video'] for z in manifest],DEST/'all_14_examples.mp4')
    selected=[next(z for z in manifest if z['case_number']==n) for n in SHORTLIST]
    concat([DEST/z['video'] for z in selected],DEST/'shortlist_6.mp4')
    contact(selected,DEST/'shortlist_contact.jpg')
    for p in range(3):contact(manifest[p*6:(p+1)*6],DEST/f'all_contact_{p+1}.jpg')
    index(manifest)
    assert all(sha(OUT/n)==h for n,h in input_hash.items())
    assert all(sha(SRC/sid/'sam2_raw_v1/masks_rle.jsonl.gz')==h for sid,h in mask_hash.items())
    verified=[]
    for p,expected in [(DEST/z['video'],80) for z in manifest]+[(DEST/'all_14_examples.mp4',1120),(DEST/'shortlist_6.mp4',480)]:
        cap=cv2.VideoCapture(str(p));n=0
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==10
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(H,W,3);n+=1
        cap.release();assert n==expected,(p,n,expected)
        verified.append({'file':str(p.relative_to(DEST)),'decoded_frames':n})
    write(DEST/'verification.json',{'all_videos_fully_decoded':verified,'evaluation_files_unchanged':input_hash,
      'mask_sources_unchanged':mask_hash,'mask_frames_shown':len(masks),'example_cutoffs_unchanged':True,
      'new_forecasts_computed':False,'only_presentation_shortlist_selected':SHORTLIST})
    print('VERIFIED',len(verified),'VIDEOS',flush=True)

if __name__=='__main__':main()
