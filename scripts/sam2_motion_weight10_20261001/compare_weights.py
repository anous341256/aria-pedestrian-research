"""Preserved V1 versus lower motion weight: synchronized outputs and decision deltas."""
import gzip
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from run import ROOT,OUT,PARENT,VIDEO,decode
from render import overlay,load_archive,FONTS,COLORS,dashed
from common import encoder,finish_encoder,write_json


def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        return {r['frame']:r for r in map(json.loads,handle)}


def comparison_frame(bgr,f,frames,masks):
    canvas=np.full((792,1600,3),22,np.uint8)
    for n,key in enumerate(['weight20','weight10']):
        canvas[64:664,n*800:(n+1)*800]=overlay(bgr,masks[key],frames[key]['branches']['motion_with_camera']['decisions'],800)
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
    for n,(key,title) in enumerate([('weight20','原版：运动评分 20%'),('weight10','新版：运动评分 10%')]):
        x=n*800+12;decisions=frames[key]['branches']['motion_with_camera']['decisions']
        d.text((x,12),title,font=FONTS[28],fill='white')
        visible=' '.join(i for i,m in masks[key].items() if m.any()) or '无'
        memory=' '.join(i for i,c in decisions.items() if c['memory_written']) or '无'
        d.text((x,674),f'原片 {f/10:.1f} 秒   输出：{visible}   写记忆：{memory}',font=FONTS[20],fill='white')
    d.text((12,714),'两版均使用相机旋转；只改变评分权重，框扩张规则和位置硬阈值保持不变。',font=FONTS[20],fill=(225,225,225))
    d.text((12,748),'虚线为预测框；实际 SAM2 提示框还会扩大。输出多少不代表身份是否正确。',font=FONTS[20],fill=(190,190,190))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def case_frame(bgr,f,identity,rows,finals,candidates):
    canvas=np.full((1260,1920,3),22,np.uint8)
    for n,(key,title) in enumerate([('weight20','原版 20%'),('weight10','新版 10%')]):
        block=rows[key][f]['branches']['motion_with_camera'];decision=block['decisions'][identity]
        prior=decision['prior'];empty=np.zeros((1920,2560),bool)
        c=candidates[key].get((f,identity+'_motion_box'))
        guided=decode(c) if c is not None else empty
        current=decode(finals[key][(f,identity)])
        for j,(mask,heading) in enumerate([(empty,'预测框与实际提示框'),(guided,'运动提示产生的候选'),(current,'最终接收结果')]):
            tile=overlay(bgr,{identity:mask},{identity:decision},640)
            if j==0 and prior and prior.get('search_box') is not None:dashed(tile,prior['search_box'],(240,240,240),.25)
            y=n*630;canvas[y+48:y+528,j*640:(j+1)*640]=tile
        image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image)
        for j,heading in enumerate(['预测框与实际提示框','运动提示产生的候选','最终接收结果']):
            d.text((j*640+10,n*630+8),title+' / '+heading,font=FONTS[22],fill='white')
        score=decision.get('score');score='无' if score is None else f'{score:.3f}'
        p=prior.get('search_box') if prior else None
        area=0 if p is None else (p[2]-p[0])*(p[3]-p[1])/(2560*1920)
        d.text((12,n*630+540),f'{identity}  原片 {f/10:.1f} 秒   评分 {score}   写记忆：{decision["memory_written"]}   提示框占画面：{area:.1%}',font=FONTS[22],fill='white')
        d.text((12,n*630+577),'彩色虚线：位置预测；白色虚线：实际 SAM2 框提示。候选和结果未经人工修正。',font=FONTS[20],fill=(200,200,200))
        canvas=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    return canvas


def run():
    paths={'weight20':PARENT,'weight10':OUT}
    rows={key:records(path/'decisions.jsonl.gz') for key,path in paths.items()}
    finals={key:load_archive(path/'masks_rle.jsonl.gz',['frame','id'],lambda r:r['branch']=='motion_with_camera') for key,path in paths.items()}
    summary={};changes=[];first=None
    for key in paths:
        counts=Counter();writes=Counter();statuses=Counter()
        for f,row in rows[key].items():
            for identity,c in row['branches']['motion_with_camera']['decisions'].items():
                counts[identity]+=int(c['selected_candidate'] is not None)
                writes[identity]+=int(c['memory_written']);statuses[c['status']]+=1
        summary[key]=dict(nonempty_masks=dict(counts),memory_writes_including_initial=dict(writes),statuses=dict(statuses))
    for f in range(100,340):
        for identity in 'ABCDE':
            a=rows['weight20'][f]['branches']['motion_with_camera']['decisions'][identity]
            b=rows['weight10'][f]['branches']['motion_with_camera']['decisions'][identity]
            mask_changed=finals['weight20'][(f,identity)]!=finals['weight10'][(f,identity)]
            decision_changed=any(a.get(k)!=b.get(k) for k in ['selected_candidate','memory_written','reference_promoted','status'])
            if mask_changed or decision_changed:
                changes.append(dict(frame=f,id=identity,mask_changed=mask_changed,decision_changed=decision_changed))
                if first is None:first=changes[-1]
    cases=[(160,'E'),(190,'B'),(220,'C'),(240,'C'),(268,'A'),(270,'A'),(270,'D'),(330,'B')]
    candidates={key:load_archive(path/'candidates_rle.jsonl.gz',['frame','candidate_id'],lambda r:r['branch']=='motion_with_camera' and r['frame'] in {f for f,_ in cases}) for key,path in paths.items()}
    case_logs=[]
    for f,identity in cases:
        entry=dict(frame=f,id=identity,versions={})
        for key in paths:
            block=rows[key][f]['branches']['motion_with_camera'];decision=block['decisions'][identity]
            entry['versions'][key]=dict(decision=decision,candidates=[c for c in block['candidates'] if c['owner']==identity])
        case_logs.append(entry)
    frozen=[]
    for f,identity in [(268,'A'),(270,'A'),(240,'C')]:
        block=rows['weight20'][f]['branches']['motion_with_camera'];decision=block['decisions'][identity]
        c=next(c for c in block['candidates'] if c['candidate_id']==decision['selected_candidate'])
        movement=float(np.exp(-.125*min(c['innovation']['mahalanobis2'],100)))
        score=.50625*c['appearance']+.28125*np.clip(c['iou_score'],0,1)+.1*movement+.1125*c['presence']
        frozen.append(dict(frame=f,id=identity,old_score=c['score'],score10_on_same_saved_candidate=float(score),passes_score_only_for_memory=bool(score>=.76)))
    dest=OUT/'presentation';dest.mkdir(exist_ok=True)
    enc,log=encoder(dest/'weight20_vs_weight10.mp4',1600,792,10,19)
    cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    for f in range(100,340):
        ok,bgr=cap.read();assert ok
        masks={key:{i:decode(finals[key][(f,i)]) for i in 'ABCDE'} for key in paths}
        rendered=comparison_frame(bgr,f,{key:rows[key][f] for key in paths},masks)
        enc.stdin.write(rendered.tobytes())
        if f%10==0 or f in [268,339]:cv2.imwrite(str(dest/f'weights_{f:04d}.jpg'),rendered,[cv2.IMWRITE_JPEG_QUALITY,95])
        for cf,identity in cases:
            if f==cf:cv2.imwrite(str(dest/f'weight_case_{identity}_{f:04d}.jpg'),case_frame(bgr,f,identity,rows,finals,candidates),[cv2.IMWRITE_JPEG_QUALITY,96])
        if (f-99)%60==0:print(f'Weight comparison rendered {f-99}/240',flush=True)
    cap.release();finish_encoder(enc,log)
    write_json(OUT/'weight_comparison.json',dict(summary=summary,first_changed_output_or_decision=first,changed_entries=changes,frozen_candidate_rescoring=frozen,cases=case_logs,
        interpretation='Counts are coverage, not accuracy. Full rerun includes accumulated state/memory feedback; frozen rescoring holds the historical candidates fixed.'))
    print(json.dumps(dict(summary=summary,first_change=first,frozen_candidate_rescoring=frozen),indent=2),flush=True)


if __name__=='__main__':run()
