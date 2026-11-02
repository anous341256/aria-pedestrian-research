"""Display policy output separately from tentative evidence and active prompts."""
import gzip,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from run import ROOT,OUT,OLD,VIDEO,BRANCH,decode
from visuals import overlay,dashed,load_archive,FONTS,COLORS,NAMES,words
from common import encoder,finish_encoder,write_json,digest

LABELS={'INITIALIZED':'初始标注','TRACKED':'策略接收','RECOVERED':'多帧确认后恢复','PENDING':'身份待确认','LOST':'未接收候选'}


def logs(path):
    with gzip.open(path,'rt',encoding='utf-8') as h:return {r['frame']:r for r in map(json.loads,h)}


def current_overlay(bgr,masks,decisions,candidates,width=1280):
    result=overlay(bgr,masks,width=width)
    for i,c in decisions.items():
        policy=c.get('prompt_policy')
        if policy and policy['enabled']:dashed(result,policy['box'],COLORS[i],width/2560)
        if c['status']=='PENDING' and c['tentative_candidate']:
            mask=candidates.get(c['tentative_candidate'])
            if mask is not None:
                m=cv2.resize(mask.astype(np.uint8),(width,width*3//4),interpolation=cv2.INTER_NEAREST)
                contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result,contours,-1,COLORS[i],1)
                xy=np.argwhere(m)
                if len(xy):
                    y,x=xy.mean(axis=0).astype(int);cv2.putText(result,'?'+i,(max(5,x),max(25,y)),cv2.FONT_HERSHEY_SIMPLEX,.65,COLORS[i],2)
    return result


def primary(bgr,f,masks,row,candidates):
    ds=row['branches'][BRANCH]['decisions'];canvas=np.full((1088,1700,3),22,np.uint8)
    canvas[64:1024,:1280]=current_overlay(bgr,masks,ds,candidates)
    im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
    d.text((15,13),'SAM2 可靠性先验：候选、身份、运动与记忆分开处理',font=FONTS[26],fill='white')
    d.text((1300,18),f'原片 {f/10:.1f} 秒',font=FONTS[22],fill='white')
    for n,i in enumerate('ABCDE'):
        y=76+n*182;c=ds[i];color=tuple(reversed(COLORS[i]));d.rectangle((1290,y,1690,y+172),fill=(34,34,34))
        d.text((1302,y+4),i+' '+NAMES[i],font=FONTS[22],fill=color)
        d.text((1302,y+34),LABELS[c['status']],font=FONTS[20],fill='white')
        identity='—' if c.get('identity_score') is None else f'{c["identity_score"]:.2f}'
        r=c.get('motion_reliability');r='—' if r is None else f'{r["score"]:.2f}'
        d.text((1302,y+64),f'身份支持 {identity}    运动可靠度 {r}',font=FONTS[18],fill='white')
        p=c.get('prompt_policy');action='启用' if p and p['enabled'] else '停用'
        d.text((1302,y+91),f'框提示：{action}   连续候选：{c.get("pending_count",0)}',font=FONTS[18],fill='white')
        memory='写入' if c['memory_written'] else '保持';motion='更新' if c.get('motion_updated') else '仅预测'
        d.text((1302,y+117),f'记忆：{memory}    主运动：{motion}',font=FONTS[18],fill='white')
        source=c.get('selected_candidate') or c.get('tentative_candidate') or ''
        source='运动提示' if source.endswith('motion_box') else '记忆传播' if source.endswith('memory') else '无 / 初始'
        d.text((1302,y+144),'候选来源：'+source,font=FONTS[16],fill=(205,205,205))
    d.text((15,1034),'实色 mask：策略接收    细轮廓与 ?：待确认    虚线框：实际启用的 SAM2 提示',font=FONTS[20],fill='white')
    d.text((15,1061),'停用的预测框不在主图绘制，仍完整记录。先验分数不是准确概率；使用离线 Aria 参考旋转。',font=FONTS[18],fill=(195,195,195))
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)


def comparison(bgr,f,masks,rows):
    keys=['v1_20','v1_10','reliability'];titles=['原版 20%','降低运动权重 10%','可靠性先验组合']
    canvas=np.full((648,1920,3),22,np.uint8)
    for n,key in enumerate(keys):canvas[60:540,n*640:(n+1)*640]=overlay(bgr,masks[key],width=640)
    im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
    for n,key in enumerate(keys):
        x=n*640+12;ds=rows[key][f]['branches'][BRANCH]['decisions']
        d.text((x,14),titles[n],font=FONTS[24],fill='white')
        shown=' '.join(i for i,m in masks[key].items() if m.any()) or '无'
        mem=' '.join(i for i,c in ds.items() if c['memory_written']) or '无'
        d.text((x,550),f'原片 {f/10:.1f} 秒  输出：{shown}',font=FONTS[20],fill='white')
        d.text((x,581),'写记忆：'+mem,font=FONTS[18],fill=(220,220,220))
    d.text((12,616),'三路只画最终接收的 mask；待确认候选另存。非空多少不等于正确率，身份错误按原样保留。',font=FONTS[18],fill=(205,205,205))
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)


def sheet(bgr,f,identity,rows,finals,candidates):
    canvas=np.full((1260,1920,3),22,np.uint8);empty=np.zeros((1920,2560),bool)
    for n,key in enumerate(['v1_20','reliability']):
        c=rows[key][f]['branches'][BRANCH]['decisions'][identity];p=c.get('prior');title='原版 20%' if n==0 else '可靠性先验'
        raw=candidates[key].get((f,identity+'_memory'))
        raw=decode(raw) if raw is not None else empty
        final=decode(finals[key][(f,identity)])
        for j,m in enumerate([empty,raw,final]):
            tile=overlay(bgr,{identity:m},width=640)
            if j==0 and p:
                if p.get('box') is not None:dashed(tile,p['box'],(130,130,130),.25)
                if p.get('search_box') is not None:dashed(tile,p['search_box'],COLORS[identity],.25)
            canvas[n*630+48:n*630+528,j*640:(j+1)*640]=tile
        im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
        for j,label in enumerate(['预测与实际框提示','无运动框的记忆候选','最终接收的 mask']):d.text((j*640+10,n*630+8),title+' / '+label,font=FONTS[22],fill='white')
        status=LABELS.get(c['status'],c['status']);r=c.get('motion_reliability');r='—' if r is None else f'{r["score"]:.3f}'
        d.text((12,n*630+540),f'{identity}  原片 {f/10:.1f} 秒   {status}   写记忆：{c["memory_written"]}   运动可靠度：{r}',font=FONTS[22],fill='white')
        d.text((12,n*630+578),'灰框为预测，彩色框为实际提示；新版停用提示时没有彩色框。规则接收不代表真实身份正确。',font=FONTS[20],fill=(190,190,190))
        canvas=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
    return canvas


def run():
    paths={'v1_20':ROOT/'outputs/sam2_motion_tracker_v1_20261001','v1_10':ROOT/'outputs/sam2_motion_weight10_20261001','reliability':OUT}
    rows={k:logs(p/'decisions.jsonl.gz') for k,p in paths.items()}
    finals={k:load_archive(p/'masks_rle.jsonl.gz',['frame','id'],lambda r:r['branch']==BRANCH) for k,p in paths.items()}
    current=load_archive(OUT/'candidates_rle.jsonl.gz',['frame','candidate_id'])
    cases=[(160,'E'),(190,'B'),(191,'C'),(220,'C'),(240,'C'),(268,'A'),(270,'A'),(270,'D'),(330,'B')]
    case_candidates={'reliability':current,'v1_20':load_archive(paths['v1_20']/'candidates_rle.jsonl.gz',['frame','candidate_id'],lambda r:r['branch']==BRANCH and r['frame'] in {f for f,_ in cases})}
    dest=OUT/'presentation';dest.mkdir(exist_ok=True)
    a,al=encoder(dest/'tracking_reliability.mp4',1700,1088,10,19);b,bl=encoder(dest/'three_versions.mp4',1920,648,10,19)
    cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    for f in range(100,340):
        ok,frame=cap.read();assert ok
        mm={key:{i:decode(finals[key][(f,i)]) for i in 'ABCDE'} for key in paths}
        needed=[c.get('tentative_candidate') for c in rows['reliability'][f]['branches'][BRANCH]['decisions'].values() if c['status']=='PENDING']
        pending={i:decode(current[(f,i)]) for i in needed if i is not None}
        main=primary(frame,f,mm['reliability'],rows['reliability'][f],pending);compare=comparison(frame,f,mm,rows)
        a.stdin.write(main.tobytes());b.stdin.write(compare.tobytes())
        if f%10==0 or f in [191,193,268,339]:
            cv2.imwrite(str(dest/f'diagnostic_{f:04d}.jpg'),main,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(dest/f'comparison_{f:04d}.jpg'),compare,[cv2.IMWRITE_JPEG_QUALITY,95])
        for cf,i in cases:
            if f==cf:cv2.imwrite(str(dest/f'case_{i}_{f:04d}.jpg'),sheet(frame,f,i,rows,finals,case_candidates),[cv2.IMWRITE_JPEG_QUALITY,96])
        if (f-99)%60==0:print(f'Rendered reliability {f-99}/240',flush=True)
    cap.release();finish_encoder(a,al);finish_encoder(b,bl)
    counts={}
    for key,rr in rows.items():
        n=Counter();m=Counter();statuses=Counter()
        for row in rr.values():
            for i,c in row['branches'][BRANCH]['decisions'].items():
                n[i]+=int(c['selected_candidate'] is not None);m[i]+=int(c['memory_written']);statuses[c['status']]+=1
        counts[key]=dict(nonempty_masks=dict(n),memory_writes_including_initial=dict(m),statuses=dict(statuses))
    evidence=[]
    for f,i in cases:
        evidence.append(dict(frame=f,id=i,versions={k:dict(decision=rr[f]['branches'][BRANCH]['decisions'][i],
            candidates=[c for c in rr[f]['branches'][BRANCH]['candidates'] if c['owner']==i]) for k,rr in rows.items()}))
    write_json(OUT/'comparison.json',dict(counts=counts,cases=evidence,interpretation='Output coverage and decision traces; not ground-truth accuracy.'))
    write_json(dest/'manifest.json',dict(frames=240,primary='tracking_reliability.mp4',comparison='three_versions.mp4',
        cases=[dict(frame=f,id=i) for f,i in cases],renderer_sha256=digest(Path(__file__))))
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':run()
