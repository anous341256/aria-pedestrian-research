"""Render prediction/observation decisions and inspectable candidate evidence."""
import gzip
import json
from pathlib import Path
import textwrap
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run import ROOT,OUT,OLD,VIDEO,BRANCHES,decode
from common import read_json,write_json,encoder,finish_encoder,digest

COLORS={'A':(255,210,0),'B':(0,220,255),'C':(75,230,100),'D':(220,95,235),'E':(0,130,255)}
NAMES={'A':'黑色上衣','B':'灰格纹衬衫','C':'白色层叠裙','D':'远处深色长装','E':'远处浅衣短裤'}
STATUS={'INITIALIZED':'初始标注','TRACKED':'接收观测并写记忆','OBSERVATION_UNCERTAIN':'接收观测，暂不写记忆',
        'UNMATCHED':'未匹配到可信候选','OUT_OF_VIEW_PRED':'预测位于视野外'}
REASONS={'reviewed_initial_mask':'已核查的初始人物','accepted_visual_observation':'通过当前规则',
 'no_usable_mask':'未产生可用 mask','low_mask_score':'分割评分低','low_object_presence':'目标存在评分低',
 'appearance_mismatch':'外观与参考不一致','observation_projection_invalid':'观测无法投到参考朝向',
 'motion_gate':'位置偏离预测范围','scale_gate':'大小变化过大','low_match_score':'综合评分不足',
 'candidate_claimed_by_another_id':'与另一身份争用同一候选','no_accepted_observation':'没有接收观测',
 'match_not_strong_enough':'匹配评分不足以写记忆','appearance_not_strong_enough':'外观可信度不足',
 'motion_not_strong_enough':'运动一致性不足','identity_ambiguous':'其他身份也可能匹配'}
FONTS={n:ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',n) for n in [16,18,20,22,24,26,28]}


def load_archive(path,key_fields,where=lambda r:True):
    result={}
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if where(row):result[tuple(row[k] for k in key_fields)]=row['rle']
    return result


def dashed(image,box,color,scale):
    p=np.array(box,dtype=float)*scale
    if not np.isfinite(p).all():return
    h,w=image.shape[:2]
    if p[2]<0 or p[3]<0 or p[0]>=w or p[1]>=h:return
    p[[0,2]]=np.clip(p[[0,2]],1,w-2);p[[1,3]]=np.clip(p[[1,3]],1,h-2)
    x0,y0,x1,y1=np.round(p).astype(int)
    for y in (y0,y1):
        for x in range(x0,x1,18):cv2.line(image,(x,y),(min(x+10,x1),y),color,2)
    for x in (x0,x1):
        for y in range(y0,y1,18):cv2.line(image,(x,y),(x,min(y+10,y1)),color,2)


def overlay(bgr,masks,decisions=None,width=1280):
    height=width*3//4
    image=cv2.resize(bgr,(width,height),interpolation=cv2.INTER_AREA)
    small={k:cv2.resize(m.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
           for k,m in masks.items()}
    counts=sum(m.astype(np.uint8) for m in small.values())
    for identity,mask in small.items():
        region=mask&(counts==1)
        image[region]=(.66*image[region]+.34*np.array(COLORS[identity])).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,COLORS[identity],2)
    yy,xx=np.indices((height,width));image[(counts>1)&((xx+yy)%12<3)]=235
    for identity,mask in small.items():
        decision=decisions.get(identity) if decisions else None
        prior=decision.get('prior') if decision else None
        center=None
        if prior and prior['box'] is not None:
            dashed(image,prior['box'],COLORS[identity],width/2560)
            center=(np.array(prior['box'][:2])+prior['box'][2:])/2*width/2560
            if 0<=center[0]<width and 0<=center[1]<height:
                cv2.drawMarker(image,tuple(np.round(center).astype(int)),COLORS[identity],cv2.MARKER_TILTED_CROSS,12,2)
        if mask.any():
            dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
            _,_,_,(x,y)=cv2.minMaxLoc(dist)
            dx={'A':-42,'B':45,'C':35,'D':-35,'E':30}[identity]
            lx=max(5,min(width-28,x+dx));ly=max(24,min(height-6,y-22))
            cv2.line(image,(x,y),(lx+8,ly-7),COLORS[identity],1,cv2.LINE_AA)
            cv2.rectangle(image,(lx-2,ly-22),(lx+23,ly+3),(15,15,15),-1)
            cv2.putText(image,identity,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,.73,COLORS[identity],2,cv2.LINE_AA)
            if center is not None and decision and decision.get('observation_box') is not None:
                obs=np.asarray(decision['observation_box'])
                observed=(obs[:2]+obs[2:])/2*width/2560
                if np.max(np.abs(center))<100000:
                    cv2.line(image,tuple(np.clip(np.round(center),-10000,10000).astype(int)),
                             tuple(np.round(observed).astype(int)),COLORS[identity],1,cv2.LINE_AA)
    return image


def words(draw,xy,value,max_width,font,color=(230,230,230),max_lines=2):
    line='';rows=[]
    for char in value:
        if draw.textlength(line+char,font=font)>max_width and line:
            rows.append(line);line=char
        else:line+=char
    if line:rows.append(line)
    for i,line in enumerate(rows[:max_lines]):draw.text((xy[0],xy[1]+i*24),line,font=font,fill=color)


def primary(bgr,masks,row):
    decisions=row['branches']['motion_with_camera']['decisions']
    canvas=np.full((1088,1700,3),22,np.uint8)
    canvas[64:1024,:1280]=overlay(bgr,masks,decisions)
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
    d.text((15,13),'SAM2 运动评分 10%  相机旋转辅助跟踪',font=FONTS[28],fill='white')
    d.text((1010,19),f'原片 {row["source_seconds"]:.1f} 秒  第 {row["frame"]} 帧',font=FONTS[22],fill='white')
    for n,identity in enumerate('ABCDE'):
        y=75+n*183;c=decisions[identity];color=tuple(reversed(COLORS[identity]))
        d.rectangle((1290,y,1690,y+174),fill=(34,34,34))
        d.text((1302,y+5),f'{identity}  {NAMES[identity]}',font=FONTS[24],fill=color)
        d.text((1302,y+36),STATUS[c['status']],font=FONTS[20],fill='white')
        source=c.get('selected_candidate') or '无'
        source='运动提示' if source.endswith('motion_box') else '记忆传播' if source.endswith('memory') else '初始' if source=='initial' else source
        score='—' if c.get('score') is None else f'{c["score"]:.2f}'
        d.text((1302,y+63),f'候选：{source}    匹配评分：{score}',font=FONTS[18],fill=(220,220,220))
        innovation='—' if c.get('innovation_px') is None else f'{c["innovation_px"]:.0f} px'
        d.text((1302,y+87),'预测与观测差距：'+innovation,font=FONTS[18],fill=(220,220,220))
        memory=c['memory_after'];action='写入' if c['memory_written'] else '保持'
        d.text((1302,y+111),f'记忆：{action}  参考 {len(memory["references"])} / 近期 {len(memory["recent"])}',font=FONTS[18],fill=(220,220,220))
        reason='；'.join(REASONS.get(s,s) for s in c['reason'].split(';'))
        words(d,(1302,y+137),reason,375,FONTS[16],(200,200,200),max_lines=1)
    d.text((15,1036),'虚线框：一步预测    实色 mask：接收的视觉观测    连线：预测与观测的差距',font=FONTS[20],fill='white')
    d.text((15,1062),'使用离线 Aria 参考旋转；评分与规则解释不等于真实准确率。保留失败输出，未人工补齐。',font=FONTS[18],fill=(185,185,185))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def comparison(bgr,base,branches,row):
    images=[overlay(bgr,base,width=640)]+[overlay(bgr,branches[b],row['branches'][b]['decisions'],640) for b in BRANCHES]
    canvas=np.full((648,1920,3),22,np.uint8)
    canvas[60:540]=np.hstack(images)
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
    titles=['原始 SAM2','运动权重 10%  不补偿旋转','运动权重 10%  补偿相机旋转']
    for i,title in enumerate(titles):
        x=i*640+12
        d.text((x,14),title,font=FONTS[22],fill='white')
        d.text((x,551),f'原片 {row["source_seconds"]:.1f} 秒 / 第 {row["frame"]} 帧',font=FONTS[20],fill='white')
        if i:
            states=row['branches'][BRANCHES[i-1]]['decisions']
            shown=' '.join(k for k,m in branches[BRANCHES[i-1]].items() if m.any()) or '无'
            written=' '.join(k for k,v in states.items() if v['memory_written']) or '无'
            d.text((x,582),f'接收 mask：{shown}    写入记忆：{written}',font=FONTS[18],fill=(225,225,225))
        else:d.text((x,582),'固定五个身份，保留原始错误与重叠',font=FONTS[18],fill=(225,225,225))
    d.text((15,617),'A 青色   B 黄色   C 绿色   D 紫色   E 橙色；虚线为预测，mask 为视觉观测。两条新分支只改变旋转补偿。',font=FONTS[18],fill=(200,200,200))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def case_sheet(bgr,f,identity,row,final,candidate_masks,baseline):
    record=row['branches']['motion_with_camera']
    decision=record['decisions'][identity]
    empty=np.zeros((1920,2560),bool)
    raw=candidate_masks.get((f,'motion_with_camera',identity+'_memory'))
    guided=candidate_masks.get((f,'motion_with_camera',identity+'_motion_box'))
    masks=[None,baseline,decode(raw) if raw is not None else empty,
           decode(guided) if guided is not None else empty,empty,final]
    titles=['原图','原始 SAM2 对照','本版记忆传播候选','本版运动提示候选','一步预测区域','本版最终接收结果']
    tiles=[]
    for i,(mask,title) in enumerate(zip(masks,titles)):
        tile=np.full((548,640,3),22,np.uint8)
        tile[45:525]=overlay(bgr,{identity:empty if mask is None else mask},
            {identity:decision} if i in [3,4,5] else None,640)
        image=Image.fromarray(cv2.cvtColor(tile,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
        d.text((10,7),title,font=FONTS[22],fill='white')
        tiles.append(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
    canvas=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
    footer=np.full((152,1920,3),22,np.uint8);canvas=np.vstack([canvas,footer])
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
    d.text((15,1106),f'{identity} {NAMES[identity]}   原片 {f/10:.1f} 秒   {STATUS[decision["status"]]}',font=FONTS[26],fill='white')
    words(d,(15,1146),'规则记录：'+'；'.join(REASONS.get(s,s) for s in decision['reason'].split(';')),1880,FONTS[22],max_lines=2)
    d.text((15,1209),'此图展示候选与决策；真实身份、可见性和根因需结合原图及前后帧复核。',font=FONTS[20],fill=(195,195,195))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def run():
    assert (OUT/'runtime.json').exists()
    rows={}
    with gzip.open(OUT/'decisions.jsonl.gz','rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);rows[row['frame']]=row
    finals=load_archive(OUT/'masks_rle.jsonl.gz',['frame','branch','id'])
    old=load_archive(OLD/'masks_rle.jsonl.gz',['frame','id'],lambda r:r['variant']=='baseline')
    cases=[(110,'D'),(140,'B'),(160,'E'),(190,'B'),(191,'C'),(220,'C'),(240,'C'),(268,'A'),(270,'A'),(270,'D'),(270,'B'),(330,'B')]
    candidates=load_archive(OUT/'candidates_rle.jsonl.gz',['frame','branch','candidate_id'],
                             lambda r:r['branch']=='motion_with_camera' and r['frame'] in {f for f,_ in cases})
    dest=OUT/'presentation';dest.mkdir(exist_ok=True)
    primary_encoder,primary_log=encoder(dest/'tracking_diagnostics.mp4',1700,1088,10,19)
    compare_encoder,compare_log=encoder(dest/'three_way_comparison.mp4',1920,648,10,19)
    cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    for f in range(100,340):
        ok,bgr=cap.read();assert ok
        branches={b:{identity:decode(finals[(f,b,identity)]) for identity in 'ABCDE'} for b in BRANCHES}
        baseline={identity:decode(old[(f,identity)]) for identity in 'ABCDE'}
        main=primary(bgr,branches['motion_with_camera'],rows[f])
        compare=comparison(bgr,baseline,branches,rows[f])
        primary_encoder.stdin.write(main.tobytes());compare_encoder.stdin.write(compare.tobytes())
        if f%10==0 or f in [181,189,191,193,339]:
            cv2.imwrite(str(dest/f'diagnostic_{f:04d}.jpg'),main,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(dest/f'comparison_{f:04d}.jpg'),compare,[cv2.IMWRITE_JPEG_QUALITY,95])
        for cf,identity in cases:
            if f==cf:
                sheet=case_sheet(bgr,f,identity,rows[f],branches['motion_with_camera'][identity],
                                 candidates,baseline[identity])
                cv2.imwrite(str(dest/f'case_{identity}_{f:04d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,96])
        if (f-99)%60==0:print(f'Rendered {f-99}/240',flush=True)
    cap.release();finish_encoder(primary_encoder,primary_log);finish_encoder(compare_encoder,compare_log)
    write_json(dest/'manifest.json',dict(frames=240,primary='tracking_diagnostics.mp4',
        comparison='three_way_comparison.mp4',candidate_sheets=[dict(frame=f,id=i) for f,i in cases],
        renderer_sha256=digest(Path(__file__)),note='All annotations are unchanged model outputs.'))
    print('Diagnostics and comparison videos rendered.',flush=True)


if __name__=='__main__':run()
