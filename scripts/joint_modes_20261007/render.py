"""Fixed original cases, six synchronized panels, explicit mode selection."""
from pathlib import Path
import json, gzip, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/joint_modes_20261007';PRES=OUT/'presentation'
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
BASE=ROOT/'outputs/factorized_forecast_v1_20261002';ATT=ROOT/'outputs/transformer_fusion_20261005'
FFMPEG=Path('E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
FONT='C:/Windows/Fonts/msyh.ttc'
COLORS=[(235,180,50),(65,190,245),(185,100,240)]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def caption(im,texts):
    canvas=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(canvas)
    for x,y,s,size in texts:d.text((x,y),s,font=ImageFont.truetype(FONT,size),fill='white')
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def draw_point(panel,xy,color,radius=8):
    p=np.asarray(xy)*.25
    if not np.isfinite(p).all():return
    if np.all(p>=0) and np.all(p<[640,480]):cv2.circle(panel,tuple(p.astype(int)),radius,color,2,cv2.LINE_AA)
    else:cv2.drawMarker(panel,tuple(np.clip(p,[8,8],[631,471]).astype(int)),color,cv2.MARKER_TILTED_CROSS,14,2)


def draw_path(panel,path,color,width=2):
    p=np.asarray(path)*.25
    if not np.isfinite(p).all():return
    p=np.clip(p,-10000,10000).astype(np.int32)
    for a,b in zip(p[:-1],p[1:]):cv2.line(panel,tuple(a),tuple(b),color,width,cv2.LINE_AA)


def masks(cid):
    path=ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:records=[json.loads(s) for s in f]
    return {(r['id'],r['frame']):r for r in records}


def mask_array(rle):
    x1,y1,x2,y2=map(int,rle[-4:]);counts=np.asarray(rle[:-4],np.int64)
    crop=np.repeat(np.arange(len(counts))%2,counts).reshape(y2-y1+1,x2-x1+1).astype(np.uint8)
    a=np.zeros((1920,2560),np.uint8);a[y1:y2+1,x1:x2+1]=crop
    return cv2.resize(a,(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)


def load_predictions():
    predictions={};meta=read(OLD/'samples.json');data=dict(np.load(OLD/'dataset.npz'))
    for f in read(OLD/'folds.json'):
        n=f['fold'];idx=f['test_indices']
        old=np.load(BASE/f'fold_{n}/seed_17/separated_dual/predictions_reviewed.npz')
        a=np.load(ATT/f'fold_{n}/seed_17/A_early_fusion/predictions_reviewed.npz')
        d=np.load(ATT/f'fold_{n}/seed_17/D_bidirectional/predictions_reviewed.npz')
        one=np.load(OUT/f'fold_{n}/seed_17/K1/predictions_test.npz')
        joint=np.load(OUT/f'fold_{n}/seed_17/K3/predictions_test.npz')
        for j,i in enumerate(idx):
            predictions[meta[i]['key']]=dict(index=i,old=old['full'][j],A=a['pred'][j],D=d['pred'][j],K1=one['pred'][j],
                K3=joint['pred'][j],all=joint['image_candidates'][j],selected=int(joint['selected'][j]),
                scores=joint['group_logits'][joint['groups'][j]],camera=joint['camera_candidates'][j],component=joint['component_candidates'][j])
    return data,meta,predictions


def plots(data,meta,predictions,keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    fig,axes=plt.subplots(len(keys),3,figsize=(14,3.2*len(keys)),squeeze=False,constrained_layout=True)
    for row,key in enumerate(keys):
        z=predictions[key];i=z['index'];chosen=z['selected'];colors=['#32abd8','#f4a641','#b86aca']
        rv=data['cam'][i,6:];gtangle=np.rad2deg(Rotation.from_rotvec(rv).magnitude())
        comp=data['xy'][i,6:];H=data['H'][i,6:];hom=np.concatenate([comp,np.ones((15,1))],-1)
        q=np.einsum('tij,tj->ti',H,hom);stable=q[:,:2]/q[:,2:]
        axes[row,0].plot(np.arange(1,16)/10,gtangle,'k--',label='Reference')
        axes[row,1].plot(stable[:,0],stable[:,1],'k--',label='Reference')
        axes[row,2].plot(comp[:,0],comp[:,1],'k--',label='Reference')
        for k in range(3):
            width=2.6 if k==chosen else 1.;style='-' if k==chosen else ':'
            label=f'Mode {k+1}'+(' SELECTED' if k==chosen else '')
            axes[row,0].plot(np.arange(1,16)/10,np.rad2deg(Rotation.from_rotvec(z['camera'][k]).magnitude()),style,color=colors[k],lw=width,label=label)
            axes[row,1].plot(*z['component'][k].T,style,color=colors[k],lw=width,label=label)
            axes[row,2].plot(*z['all'][k].T,style,color=colors[k],lw=width,label=label)
        for col,title in enumerate(['Camera rotation magnitude (not direction)','Pedestrian residual / fixed orientation','Composed future image position']):
            ax=axes[row,col];ax.set_title(f'{key}\n{title}',fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7)
            if col:ax.invert_yaxis();ax.set_xlabel('x / px');ax.set_ylabel('y / px');ax.set_aspect('equal',adjustable='datalim')
            else:ax.set_xlabel('Future seconds');ax.set_ylabel('degrees')
    fig.savefig(PRES/'fixed_branch_outputs.png',dpi=150);fig.savefig(PRES/'fixed_branch_outputs.pdf');plt.close(fig)
    summary=read(OUT/'summary.json');names=['hold_last','old_GRU_flow','A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry','joint_K1','joint_K3']
    labels=['Hold last','Old GRU + flow','A Early fusion / self-attn','B Person self-attn','C Separate / geometry','D Bidirectional cross-attn','E One-way cross-attn','F Cross-attn / geometry','Joint K=1 control','Joint K=3 selected']
    fig,ax=plt.subplots(figsize=(10,5.8),constrained_layout=True)
    values=[summary[n]['ADE'] for n in names]
    ax.barh(labels,values,color=['#9aa2af']*8+['#77b6ae','#e99843']);ax.invert_yaxis()
    for i,n in enumerate(names):
        ss=list(summary[n]['seed_ADE'].values());ax.scatter(ss,[i]*len(ss),color='#26374c',s=18,zorder=3)
        ax.text(max(values[i],max(ss))+2,i,f'{values[i]:.2f}',va='center')
    ax.set_xlim(0,max(max(summary[n]['seed_ADE'].values()) for n in names)*1.15);ax.set_xlabel('ADE / px (lower is better); dots = 3 training seeds')
    ax.set_title('Same 117 held-out windows | 6 recording folds\nSystem comparison; K1/K3 share additional adapter-training budget')
    ax.grid(axis='x',alpha=.2);fig.savefig(PRES/'comparison.png',dpi=160);plt.close(fig)


def main():
    PRES.mkdir(parents=True,exist_ok=True);data,meta,predictions=load_predictions()
    keys=read(OLD/'presentation/selection.json')['fixed']
    write(PRES/'selection.json',dict(fixed=keys,seed=17,rule='Same original nine fixed cases; no new selection based on errors.'))
    frames={};mm={}
    for cid in sorted({meta[predictions[key]['index']]['clip'] for key in keys}):
        cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4'));ff=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            ff.append(cv2.resize(im,(640,480)))
        cap.release();frames[cid]=ff;mm[cid]=masks(cid)
    dest=PRES/'fixed_9_comparison.mp4'
    if not dest.exists():
        with dest.with_suffix('.ffmpeg.log').open('wb') as log:
            proc=subprocess.Popen([str(FFMPEG),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1200','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=log)
            for key in keys:
                z=predictions[key];i=z['index'];m=meta[i];scores=np.exp(z['scores']-z['scores'].max());scores/=scores.sum()
                names=[('old','原 GRU＋flow'),('A','A：先融合＋self-attention'),('D','D：双向 cross-attention'),('K1','新方案：单候选对照'),('K3',f'新方案：联合选择模式 {z["selected"]+1}'),('all','三个完整未来候选（亮色为选中）')]
                for j,frame in enumerate(range(m['cutoff']-5,m['cutoff']+16)):
                    im=frames[m['clip']][frame].copy();r=mm[m['clip']].get((m['id'],frame))
                    if r:
                        mask=mask_array(r['rle']);im[mask]=im[mask]*.84+.16*np.array([255,220,80])
                    canvas=np.full((1200,1920,3),(27,23,19),np.uint8)
                    text=[(16,7,f'{key} | 固定规则案例 · 相同画面和参考 · 种子17 · 半速',27)]
                    for cell,(name,label) in enumerate(names):
                        x=(cell%3)*640;y=50+(cell//3)*552;panel=im.copy()
                        if name!='all':
                            ade=np.linalg.norm(z[name]-data['xy'][i,6:],axis=-1).mean();label+=f' | ADE {ade:.1f}'
                        text.append((x+10,y,label,22));draw_point(panel,data['xy'][i,j],(255,255,255),6)
                        if j>=6:
                            if name=='all':
                                for k in range(3):
                                    draw_path(panel,z['all'][k,:j-5],COLORS[k],3 if k==z['selected'] else 1)
                                    draw_point(panel,z['all'][k,j-6],COLORS[k],10 if k==z['selected'] else 5)
                            else:
                                draw_path(panel,z[name][:j-5],(0,170,255));draw_point(panel,z[name][j-6],(0,170,255))
                        canvas[y+42:y+522,x:x+640]=panel
                    stage='观察过去0.5秒' if j<6 else f'预测未来 {(j-5)/10:.1f} 秒（不再读取观测）'
                    text.append((16,1160,stage+f' | 模式评分：{scores[0]:.2f} / {scores[1]:.2f} / {scores[2]:.2f}（非校准概率）',24))
                    canvas=caption(canvas,text);proc.stdin.write(canvas.tobytes());proc.stdin.write(canvas.tobytes())
                    if j==20:cv2.imwrite(str(PRES/f'{key}.jpg'),canvas)
            proc.stdin.close();assert proc.wait(timeout=60)==0
    cap=cv2.VideoCapture(str(dest));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1200,1920,3);count+=1
    cap.release();assert count==len(keys)*42
    write(PRES/'video_checks.json',dict(passed=True,frames=count,seconds=count/10))
    plots(data,meta,predictions,keys[:3])
    (PRES/'START_HERE.txt').write_text('共享模式联合候选实验\n\nfixed_9_comparison.mp4：原固定九案例，六栏同步，38秒左右。\ncomparison.png：三种子117窗口总体比较。\nfixed_branch_outputs.png/pdf：前三个固定案例的相机、行人残差、最终位置候选。\n模式数字只在各折内有意义；评分不是校准概率。所有案例按旧规则固定，未按新结果挑选。\n',encoding='utf-8')
    print('Rendered and decoded',count,'frames',flush=True)


if __name__=='__main__':main()
