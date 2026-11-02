"""Readable videos and static figures; display future RGB only after inference."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from prepare import ROOT,SRC,OUT,read,write
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder

FONT=Path('C:/Windows/Fonts/msyh.ttc')
if not FONT.exists():FONT=Path('/mnt/c/Windows/Fonts/msyh.ttc')
FONTS={n:ImageFont.truetype(str(FONT),n) for n in [18,20,22,24,28,30]}
COL={'history':(80,175,255),'reference':(80,230,135),'linear':(255,140,90),'mamba_b0':(218,130,255),'stationary':(215,220,229),'cvcs':(218,187,74)}
LABEL={'stationary':'保持当前位置','linear':'匀速外推','cvcs':'最近速度+尺度外推','mamba_b0':'TrajMamba-R0 B0'}
SCENES={'street_S1':'白天街道 · S1','station_S4':'站台→扶梯 · S4','dusk_follow_S1':'傍晚人行道 · S1','dusk_cross_S4':'傍晚过街 · S4'}

def txt(draw,xy,s,size=22,fill=(231,236,244)):
    draw.text(xy,s,font=FONTS[size],fill=fill)

def line(im,pts,rgb,thick=3):
    pp=np.array(pts,dtype=float)/4
    pp=np.clip(pp,-100000,100000).astype(np.int32)
    cv2.polylines(im,[pp.reshape(-1,1,2)],False,tuple(reversed(rgb)),thick,cv2.LINE_AA)
    for point in pp[::3]:
        if 0<=point[0]<640 and 0<=point[1]<480:cv2.circle(im,tuple(point),3,tuple(reversed(rgb)),-1,cv2.LINE_AA)

def box(im,b,rgb,xywh=True,thick=2):
    b=np.array(b,dtype=float)
    if xywh:
        if (b[2:]<=0).any():return
        b=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]
    b=np.clip(b/4,-100000,100000).astype(int)
    cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),tuple(reversed(rgb)),thick,cv2.LINE_AA)

def canvas(spec,row,current,future,reveal=True):
    left=current.copy();right=future.copy()
    hist=np.array(row['history_xyxy']);centers=(hist[:,:2]+hist[:,2:])/2
    box(left,hist[-1],COL['history'],False)
    line(left,centers,COL['history'],4)
    for m in ['linear','mamba_b0']:
        pred=np.array(row['predictions'][m]);line(left,np.r_[centers[-1:],pred[:,:2]],COL[m])
        box(right,pred[-1],COL[m],thick=3)
    if reveal:
        target=np.array(row['target_xywh']);line(left,np.r_[centers[-1:],target[:,:2]],COL['reference'],2)
        box(right,target[-1],COL['reference'],thick=4)
    else:right[:]=(30,24,20)
    sheet=Image.new('RGB',(1280,800),(15,21,31));d=ImageDraw.Draw(sheet)
    txt(d,(24,12),f'{SCENES[spec["key"]]}  |  过去约 0.5 秒 → 未来 1.5 秒',28)
    txt(d,(24,53),f'预测起点 t={row["frame"]/10:.1f}s   /   图像中的位置预测',20)
    txt(d,(665,53),f'后续画面 t+1.5s={row["frame"]/10+1.5:.1f}s  ·  仅用于核对',20)
    sheet.paste(Image.fromarray(cv2.cvtColor(left,cv2.COLOR_BGR2RGB)),(0,90))
    sheet.paste(Image.fromarray(cv2.cvtColor(right,cv2.COLOR_BGR2RGB)),(640,90))
    if not reveal:txt(d,(745,290),'后续画面暂不揭晓',30)
    for x,m,label in [(24,'history','已观察历史'),(255,'reference','后续 SAM 参考'),(544,'linear','匀速外推'),(807,'mamba_b0','已训练的 B0 网络')]:
        d.line((x,597,x+30,597),fill=COL[m],width=5);txt(d,(x+42,580),label,22)
    txt(d,(24,620),'左侧曲线：各未来帧的图像坐标，不能当作地面上的路线。',20)
    txt(d,(24,649),'右侧叠框：预测提前算好，再用 1.5 秒后的画面核对；该画面不输入模型。',20)
    if reveal:
        text='本窗口 ADE / FDE（原图像素）：  '
        for m in ['stationary','linear','mamba_b0']:
            z=row['metrics'][m];text+=f'{LABEL[m]} {z["ADE_1.5s"]:.0f}/{z["FDE_1.5s"]:.0f}   '
        txt(d,(24,688),text,18)
    else:txt(d,(24,688),'模型只收到历史坐标；本轮未在 Aria 数据上训练。',20)
    edge='本窗口存在触边框' if row['edge_truncated'] else '本窗口无触边框'
    txt(d,(24,732),f'{edge}  ·  自动 mask 外接框，非人工金标  ·  B0 为 PIE 预训练独立复现',18,(167,182,201))
    txt(d,(24,764),'越出画面的预测照常计误差；无效宽高不画框。轨迹线保留原始输出。',18,(167,182,201))
    return cv2.cvtColor(np.asarray(sheet),cv2.COLOR_RGB2BGR)

def get_frames(spec):
    cap=cv2.VideoCapture(str(SRC/spec['recording']/'rgb.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,spec['start']);frames={}
    for f in range(spec['start'],spec['end']+1):
        ok,im=cap.read();assert ok
        frames[f]=cv2.resize(im,(640,480),interpolation=cv2.INTER_AREA)
    cap.release();return frames

def teaching(spec,rows,frames):
    fixed=read(OUT/'protocol.json')['representative_frames'][spec['key']]
    row=next(r for r in rows if r['frame']==fixed)
    proc,log=encoder(OUT/'00_how_prediction_works.mp4',1280,800,10)
    # A fixed cutoff: reveal future only after the predicted path has been shown.
    for i in range(100):
        reveal=i>=50
        im=canvas(spec,row,frames[fixed],frames[fixed+15],reveal)
        pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
        d.rectangle((0,0,1280,48),fill=(24,34,51))
        title='第 1 步：只看截至 t 的历史，网络输出未来位置' if not reveal else '第 2 步：揭晓 t+1.5 秒的画面，核对提前预测的框'
        txt(d,(24,8),title,28)
        proc.stdin.write(cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR).tobytes())
    finish_encoder(proc,log)

def plots():
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    s=read(OUT/'summary.json');methods=['stationary','linear','cvcs','mamba_b0'];keys=list(SCENES)
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),layout='constrained')
    xx=np.arange(4)
    for j,m in enumerate(methods):
        vals=[s['by_track'][k]['methods'][m]['ADE_1.5s'] for k in keys]
        bars=axes[0].bar(xx+(j-1.5)*.19,vals,.19,label=LABEL[m],color=np.array(COL[m])/255,edgecolor='#344054',linewidth=.5)
        for b in bars:axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+8,f'{b.get_height():.0f}',ha='center',fontsize=8)
        ys=[s['all']['methods'][m][f'ADE_{h:.1f}s'] for h in [.5,1,1.5]]
        axes[1].plot([.5,1,1.5],ys,'o-',label=LABEL[m],color=np.array(COL[m])/255,markeredgecolor='#344054',linewidth=2)
    axes[0].set_xticks(xx,[SCENES[k].replace(' · ','\n') for k in keys]);axes[0].set_title('4 条长轨迹：预测未来 1.5 秒')
    axes[1].set_title('205 个窗口：预测越远，误差如何变化');axes[1].set_xlabel('预测时长（秒）');axes[1].set_xticks([.5,1,1.5])
    for ax in axes:ax.set_ylabel('中心点 ADE（2560×1920 原图像素）');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9);axes[1].legend(fontsize=9)
    fig.suptitle('SAM2 自动标注上的试跑 · PIE 模型直接迁移 · 非人工金标评测',fontsize=15)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)

def main():
    specs=read(OUT/'candidates.json');manifest=[]
    for spec in specs:
        rows=read(OUT/f'{spec["key"]}_predictions.json');frames=get_frames(spec)
        dest=OUT/f'{spec["key"]}_forecast.mp4'
        proc,log=encoder(dest,1280,800,10)
        representative=read(OUT/'protocol.json')['representative_frames'][spec['key']]
        for row in rows:
            im=canvas(spec,row,frames[row['frame']],frames[row['frame']+15])
            proc.stdin.write(im.tobytes())
            if row['frame']==representative:cv2.imwrite(str(OUT/f'{spec["key"]}_example.jpg'),im)
        finish_encoder(proc,log)
        if spec['key']=='station_S4':teaching(spec,rows,frames)
        manifest.append(dict(file=dest.name,frames=len(rows),fps=10,duration_seconds=len(rows)/10,
            first_cutoff=rows[0]['frame'],last_cutoff=rows[-1]['frame'],future_offset_frames=15))
        print('RENDERED',spec['key'],len(rows),flush=True)
        del frames
    plots();write(OUT/'video_manifest.json',manifest)

if __name__=='__main__':main()
