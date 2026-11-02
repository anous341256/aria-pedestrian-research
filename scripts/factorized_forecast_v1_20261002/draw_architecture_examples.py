"""Add real, paired intermediate results beside the implemented architecture.

All curves come from one saved held-out test window. The displayed future video
and reference future poses are evaluation visuals, never model inputs.
"""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.transforms import Bbox
from PIL import Image
from draw_architecture import Architecture,COLORS,digest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/biflow_aria_v1_20261002'
RUN=ROOT/'outputs/factorized_forecast_v1_20261002'
DEST=RUN/'presentation/architecture_examples'
CASE='10_N01_020'
SEED=17
SCALE=np.array([2560.,1920.])
PAL=dict(observed='#3F719A',stable='#2E817A',predicted='#CC7D27',reference='#64717D',reconstruction='#826D94')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def warp(points,H):
    homogeneous=np.concatenate([points,np.ones(points.shape[:-1]+(1,))],axis=-1)
    q=np.einsum('...ij,...j->...i',H,homogeneous)
    return q[...,:2]/q[...,2:]


def load_case():
    metadata=read(BASE/'samples.json');index=next(i for i,m in enumerate(metadata) if m['key']==CASE)
    m=metadata[index];fold=next(f for f in read(BASE/'folds.json') if index in f['test_indices'])['fold']
    source_prediction=RUN/f'fold_{fold}/seed_{SEED}/separated_dual/predictions_reviewed.npz'
    source_camera=ROOT/f'outputs/camera_full_data_20261002/fold_{fold}/seed_{SEED}/full/predictions.npz'
    source_video=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'
    source_masks=ROOT/f'outputs/sam2_near_people_20261002/{m["clip"]}/final/masks_rle.jsonl.gz'
    data=np.load(BASE/'dataset.npz');z=np.load(source_prediction);j=list(z['indices']).index(index)
    xy=data['xy'][index];stable=warp(xy,data['H'][index]);origin=(xy[5]/SCALE).astype(np.float32)
    samples=(z['samples_normalized'][j]+origin[None,None])*SCALE
    assert np.max(abs(samples.mean(0)-z['component'][j]))<.001
    cam=np.load(source_camera)['person_rotation'][index]
    projected=[]
    for point,rotation in zip(z['component'][j],cam):
        Q=cv2.Rodrigues(rotation)[0];H=data['K'][index]@Q.T@np.linalg.inv(data['K'][index])
        projected.append(warp(point,H))
    assert np.max(abs(np.asarray(projected)-z['full'][j]))<.001
    images={};mask_arrays={}
    cap=cv2.VideoCapture(str(source_video))
    for frame in [m['cutoff'],m['cutoff']+15]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
        images[frame]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    cap.release()
    with gzip.open(source_masks,'rt',encoding='utf-8') as stream:
        for row in stream:
            r=json.loads(row)
            if r['id']!=m['id'] or r['frame'] not in images:continue
            x1,y1,x2,y2=map(int,r['rle'][-4:]);counts=np.asarray(r['rle'][:-4],np.int64)
            values=np.repeat(np.arange(len(counts))%2,counts).reshape(y2-y1+1,x2-x1+1).astype(bool)
            mm=np.zeros((1920,2560),bool);mm[y1:y2+1,x1:x2+1]=values;mask_arrays[r['frame']]=mm
    assert len(mask_arrays)==2
    # A fixed display crop is used for both images. Coordinates remain those
    # of the original 2560x1920 frame throughout computation and annotation.
    crop=[400,850,2100,1700]
    error=np.linalg.norm(z['full'][j]-xy[6:],axis=-1)
    arrays=dict(xy=xy,stable=stable,pred_component=z['component'][j],pred_image=z['full'][j],
                pred_samples=samples,reconstruction=z['reconstruction_component'][j],
                camera_reference=data['cam'][index],camera_prediction=cam,seconds=data['seconds'][index],
                K=data['K'][index],H=data['H'][index],crop=np.asarray(crop))
    np.savez_compressed(DEST/'case_arrays.npz',**arrays)
    source_paths=[BASE/'dataset.npz',BASE/'samples.json',BASE/'folds.json',source_prediction,source_camera,source_video,source_masks]
    record=dict(case=CASE,index=index,fold=fold,seed=SEED,person_id=m['id'],cutoff=m['cutoff'],
                recording=m['recording'],display_frames=[m['cutoff'],m['cutoff']+15],
                selection='Previously fixed nine-case set; one illustrative test window, not an aggregate result or best-case claim.',
                original_resolution=[2560,1920],crop_xyxy=crop,window_ADE=float(error.mean()),FDE=float(error[-1]),
                inference_uses_future_reference=False,sources={p.relative_to(ROOT).as_posix():digest(p) for p in source_paths})
    return m,arrays,images,mask_arrays,record


class Illustrated:
    def __init__(self,language,case):
        self.language=language;self.cn=language=='zh';self.m,self.a,self.images,self.masks,self.record=case
        self.diagram=Architecture(language);self.fig=self.diagram.fig
        self.fig.set_size_inches(29,17)
        self.diagram.ax.set_position([.006,.358,.716,.635])
        self.body=self.diagram.body;self.bold=self.diagram.bold;self.cards=[];self.extra_text=[]

    def ft(self,x,y,text,size=12,bold=False,color=None,ha='left',va='center',**kw):
        t=self.fig.text(x,y,text,fontproperties=self.bold if bold else self.body,fontsize=size,
                        color=color or COLORS['ink'],ha=ha,va=va,**kw)
        self.extra_text.append(t);return t

    def tag(self,x,y,label,color):
        self.fig.add_artist(Ellipse((x,y),.013,.013*29/17,transform=self.fig.transFigure,
                                   facecolor=color,edgecolor='white',linewidth=1,zorder=20))
        self.ft(x,y,label,size=10.5,bold=True,color='white',ha='center',zorder=21)

    def card(self,rect,label,title,caption,kind='person',photo=False):
        x,y,w,h=rect
        self.fig.add_artist(FancyBboxPatch((x,y),w,h,transform=self.fig.transFigure,
            boxstyle='round,pad=.0015,rounding_size=.004',edgecolor=COLORS[kind],facecolor='white',linewidth=1.1,zorder=-1))
        self.tag(x+.014,y+h-.022,label,COLORS[kind])
        self.ft(x+.027,y+h-.022,title,size=13.5,bold=True,color=COLORS[kind])
        self.ft(x+.012,y+.017,caption,size=10.3,color=COLORS['muted'])
        if photo:ax=self.fig.add_axes([x+.011,y+.045,w-.022,h-.085])
        else:ax=self.fig.add_axes([x+.035,y+.051,w-.049,h-.095])
        self.cards.append((rect,ax,label));return ax

    def style(self,ax,xy=True):
        ax.tick_params(labelsize=10,colors=COLORS['muted'])
        for s in ax.spines.values():s.set_color('#CED5DB');s.set_linewidth(.7)
        ax.grid(alpha=.23,color='#C2CCD5',linewidth=.6);ax.set_axisbelow(True)
        if xy:
            ax.set_xlabel(r'$\Delta x$ (px)',fontsize=11,labelpad=1)
            ax.set_ylabel(r'$\Delta y$ (px)',fontsize=11,labelpad=1)
            ax.invert_yaxis()
            ax.set_aspect('equal',adjustable='datalim')

    def legend(self,ax,location='best',size=9.7,ncol=1):
        leg=ax.legend(loc=location,prop=FontProperties(fname=self.body.get_file(),size=size),
                      frameon=True,framealpha=.93,edgecolor='#DFE5E8',borderpad=.35,ncol=ncol,handlelength=2)
        return leg

    def line(self,ax,xy,color,label=None,style='-',marker='o',lw=1.9,ms=3.6,alpha=1):
        origin=self.a['xy'][5]
        v=np.asarray(xy)-origin
        return ax.plot(v[:,0],v[:,1],style,color=color,marker=marker,markersize=ms,linewidth=lw,alpha=alpha,label=label)[0]

    def photo(self,ax,frame,future=False):
        rgb=self.images[frame].copy();mm=self.masks[frame]
        rgb[mm]=(rgb[mm]*.74+.26*np.array([46,160,137])).astype(np.uint8)
        ax.imshow(rgb,extent=[0,2560,1920,0],interpolation='nearest')
        x1,y1,x2,y2=self.a['crop'];ax.set_xlim(x1,x2);ax.set_ylim(y2,y1);ax.axis('off')
        if not future:
            points=self.a['xy'][:6]
            ax.plot(points[:,0],points[:,1],'-o',color='#75C9FF',lw=2.6,ms=5,markeredgecolor='#153F5F',markeredgewidth=.5)
            ax.scatter(*points[-1],s=100,facecolor='none',edgecolor='white',lw=2,zorder=8)
            ax.annotate('t',points[-1],xytext=(12,-16),textcoords='offset points',color='white',fontsize=14,weight='bold')
        else:
            ref=self.a['xy'][-1];pred=self.a['pred_image'][-1]
            ax.plot([ref[0],pred[0]],[ref[1],pred[1]],color='#F0A33E',lw=2.2)
            ax.scatter(*ref,s=110,facecolor='none',edgecolor='white',lw=2.3,zorder=8)
            ax.scatter(*pred,s=130,facecolor='none',edgecolor='#F0A33E',lw=2.5,zorder=8)
            for xy,label,color,offset in [(ref,'参考' if self.cn else 'Reference','white',(-7,-24)),
                                           (pred,'预测' if self.cn else 'Prediction','#FFC879',(8,12))]:
                ax.annotate(label,xy,xytext=offset,textcoords='offset points',fontproperties=self.bold,fontsize=12,color=color,
                            bbox=dict(boxstyle='round,pad=.25',fc='#17232D',ec='none',alpha=.7))
        label=f'{self.m["id"]} | '+('t + 1.5 s' if future else 't = 0 s')
        ax.text(.015,.97,label,transform=ax.transAxes,ha='left',va='top',fontsize=12,color='white',
                bbox=dict(boxstyle='round,pad=.3',fc='#17232D',ec='none',alpha=.78))

    def draw(self):
        cn=self.cn;d=self.diagram;d.draw()
        # Tags correspond to actual arrays, not decorative or synthesized data.
        tags=[(36,63,'A','person'),(68,63,'B','person'),(145,95,'C','camera'),
              (145,63,'D','person'),(105,24,'E','training'),(218,64,'F','geometry')]
        for x,y,label,kind in tags:
            d.ax.add_patch(Circle((x,y),1.65,facecolor=COLORS[kind],edgecolor='white',linewidth=.9,zorder=12))
            d.text(x,y,label,size=11,bold=True,color='white').set_zorder(13)
        self.ft(.754,.977,'同一真实案例的中间结果' if cn else 'Intermediate results from one real test case',size=17,bold=True)
        self.ft(.754,.949,f'{CASE}  |  seed {SEED}  |  '+('固定展示样本' if cn else 'fixed demonstration case'),size=11.5,color=COLORS['muted'])

        # B: observed versus rotation-normalized historical positions.
        ax=self.card((.746,.735,.244,.198),'B','历史轨迹如何被补偿' if cn else 'Transform the observed trajectory',
                     '蓝：原始坐标；绿：补偿坐标。相同 6 个历史时刻。' if cn else 'Same six observations; image coordinates vs. rotation-normalized coordinates.')
        self.line(ax,self.a['xy'][:6],PAL['observed'],'原始' if cn else 'Original')
        self.line(ax,self.a['stable'][:6],PAL['stable'],'旋转补偿' if cn else 'Rotation-normalized')
        ax.scatter(0,0,color=COLORS['ink'],marker='*',s=80,zorder=6,label='当前点' if cn else 'Current point')
        self.style(ax);self.legend(ax)

        # C: rotation-vector components; never mislabel these as Euler angles.
        ax=self.card((.746,.508,.244,.198),'C','预测未来相机旋转' if cn else 'Predict future camera rotations',
                     '实线：历史/预测；虚线：未来参考。单位：度。' if cn else 'Solid: history / forecast. Dashed: future reference. Degrees.',kind='camera')
        seconds=self.a['seconds'];reference=np.degrees(self.a['camera_reference']);pred=np.degrees(self.a['camera_prediction'])
        axis_colors=['#3F719A','#C88A35','#418B76']
        ax.axvspan(0,1.5,color='#F8F3E9',zorder=-3);ax.axvline(0,color='#82909A',lw=.8)
        for j,(col,name) in enumerate(zip(axis_colors,['x','y','z'])):
            ax.plot(seconds[:6],reference[:6,j],color=col,lw=2)
            ax.plot(np.r_[0,seconds[6:]],np.r_[0,pred[:,j]],color=col,lw=2,label=name)
            ax.plot(np.r_[0,seconds[6:]],np.r_[0,reference[6:,j]],color=col,lw=1.4,ls='--',alpha=.85)
        ax.set_xlim(-.5,1.5);ax.set_xlabel('Time (s)',fontsize=11,labelpad=1);ax.set_ylabel('Rotation vector (deg)',fontsize=10.5,labelpad=1)
        self.style(ax,False);self.legend(ax,ncol=3)

        # D: real per-start predictions plus their saved mean, in stabilized coordinates.
        ax=self.card((.746,.281,.244,.198),'D','预测去旋转后的行人轨迹' if cn else 'Predict the residual pedestrian trajectory',
                     '淡线：5 个起点；橙线：均值；灰虚线：未来参考。' if cn else 'Five flow starts (faint), their mean (orange), reference future (dashed).')
        self.line(ax,self.a['stable'][:6],PAL['stable'],'历史' if cn else 'History')
        for sample in self.a['pred_samples']:
            self.line(ax,np.vstack([self.a['xy'][5],sample]),PAL['predicted'],marker=None,lw=.8,alpha=.23)
        self.line(ax,np.vstack([self.a['xy'][5],self.a['pred_component']]),PAL['predicted'],'预测' if cn else 'Forecast',ms=2.8)
        self.line(ax,self.a['stable'][5:],PAL['reference'],'参考' if cn else 'Reference',style='--',marker=None,lw=1.6)
        self.style(ax);self.legend(ax,ncol=3,size=9.2)

        # E: show the actual auxiliary output without implying it denoises successfully.
        ax=self.card((.746,.054,.244,.198),'E','历史重建辅助头' if cn else 'Auxiliary history reconstruction',
                     '紫线是实际重建结果；不接入未来推理路径。' if cn else 'Actual reconstruction output; excluded from the future inference path.',kind='training')
        self.line(ax,self.a['stable'][:6],PAL['stable'],'历史输入' if cn else 'Observed history')
        self.line(ax,self.a['reconstruction'],PAL['reconstruction'],'重建' if cn else 'Reconstruction',marker='s')
        for a,b in zip(self.a['stable'][:6],self.a['reconstruction']):
            self.line(ax,np.vstack([a,b]),'#CFCDD5',marker=None,lw=.7)
        self.style(ax);self.legend(ax)

        # A/F: the same crop, real SAM2 masks, real future output, no synthetic imagery.
        ax=self.card((.035,.048,.335,.284),'A','真实观测：画面、mask 与 6 个历史点' if cn else 'Observed frame, SAM2 mask and six history points',
                     't 帧；蓝点为历史图像坐标叠加。绿色为该行人 mask。' if cn else 'Frame t; observed image coordinates overlaid. Green: the target SAM2 mask.',photo=True)
        self.photo(ax,self.m['cutoff'])
        ax=self.card((.385,.048,.335,.284),'F','重投影：回到未来画面' if cn else 'Reproject into the future image',
                     ('同一窗口：ADE %.1f px，FDE %.1f px。未来画面仅用于展示。' if cn else
                      'This window: ADE %.1f px, FDE %.1f px. Future image shown for evaluation only.')%(self.record['window_ADE'],self.record['FDE']),
                     kind='geometry',photo=True)
        self.photo(ax,self.m['cutoff']+15,future=True)
        self.ft(.035,.020,'A–F 对应模型中的同名标记；所有效果均来自同一条已保存预测。未来参考仅用于对照，不进入推理。' if cn else
                'A–F match the architecture tags. Every panel uses the same saved test prediction. Future references are for comparison only, never inference inputs.',
                size=11,color=COLORS['muted'])
        self.ft(.035,.005,'轨迹图以当前点为原点，单位为原图像素；照片采用相同裁剪窗口。' if cn else
                'Trajectory plots are centered at the current observation, in original-image pixels. Both photographs use the same crop.',
                size=10.5,color=COLORS['muted'])

    def save(self):
        self.draw();qa=self.diagram.validate()
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
        violations=[]
        for txt in self.extra_text:
            bb=txt.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            if bb.x0<0 or bb.y0<0 or bb.x1>1 or bb.y1>1:violations.append(txt.get_text())
        assert not violations,violations
        base=DEST/f'architecture_examples_{self.language}'
        for suffix in ['png','pdf','svg']:
            self.fig.savefig(base.with_suffix('.'+suffix),dpi=210,facecolor='white')
        self.fig.savefig(DEST/f'architecture_examples_{self.language}_preview.png',dpi=100,facecolor='white')
        # Export the actual panels for independent insertion into slides/docs.
        cards_dir=DEST/'panels';cards_dir.mkdir(exist_ok=True)
        tiles={}
        for (x,y,w,h),ax,label in self.cards:
            path=cards_dir/f'{label}_{self.language}.png'
            crop=Bbox.from_bounds(x*29-.10,y*17-.10,w*29+.20,h*17+.20)
            self.fig.savefig(path,dpi=180,facecolor='white',bbox_inches=crop)
            im=Image.open(path).convert('RGB');im.thumbnail((1100,550),Image.Resampling.LANCZOS)
            tile=Image.new('RGB',(1130,580),'white');tile.paste(im,((1130-im.width)//2,(580-im.height)//2));tiles[label]=tile
        grid=Image.new('RGB',(3390,1160),'white')
        for i,label in enumerate('ABCDEF'):grid.paste(tiles[label],((i%3)*1130,(i//3)*580))
        grid.save(DEST/f'example_panels_{self.language}.png')
        plt.close(self.fig)
        return qa


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    fontManager.addfont('C:/Windows/Fonts/arial.ttf')
    plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.unicode_minus':False})
    case=load_case();qa={lang:Illustrated(lang,case).save() for lang in ['zh','en']}
    record=case[-1]
    record.update(renderer='Matplotlib with actual video frames and saved prediction arrays; no image generation',
                  drawing_source=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=digest(Path(__file__))),
                  model_diagram_source=dict(path='scripts/factorized_forecast_v1_20261002/draw_architecture.py',sha256=digest(Path(__file__).with_name('draw_architecture.py'))),
                  geometry_verified=True,saved_flow_sample_mean_verified=True,layout_qa=qa,
                  exports={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
                           for p in DEST.rglob('*') if p.is_file() and (p.name.startswith(('architecture_examples_','example_panels_')) or p.parent.name=='panels')})
    for path,checksum in record['sources'].items():assert digest(ROOT/path)==checksum
    (DEST/'case_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'case':CASE,'ADE':record['window_ADE'],'FDE':record['FDE'],'directory':str(DEST),'files':list(record['exports'])},ensure_ascii=False))


if __name__=='__main__':main()
