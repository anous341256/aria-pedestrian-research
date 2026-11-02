"""Render prediction/observation decisions and inspectable candidate evidence."""
import gzip
import json
from pathlib import Path
import textwrap
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sam_io import ROOT,OLD,VIDEO,decode
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

