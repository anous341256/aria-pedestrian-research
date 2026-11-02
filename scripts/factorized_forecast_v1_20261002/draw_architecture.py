"""Publication-style vector diagrams for the implemented separated_dual model.

Uses only Matplotlib drawing primitives. Does not invoke image generation or
modify model weights, training code, or existing experiment outputs.
"""
from pathlib import Path
import json
import hashlib
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties, fontManager

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/factorized_forecast_v1_20261002/presentation/architecture'
COLORS = dict(ink='#263443', muted='#647180', line='#526171',
              camera='#3F719A', camera_fill='#F3F7FC', camera_node='#E9F1FA',
              person='#2E817A', person_fill='#F1F8F6', person_node='#E3F2EE',
              geometry='#9A7132', geometry_fill='#FBF8F1', geometry_node='#F7EDD9',
              training='#826D94', training_fill='#FAF8FC', training_node='#F1EBF7')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Architecture:
    def __init__(self, language):
        self.language = language
        self.cn = language == 'zh'
        self.fig, self.ax = plt.subplots(figsize=(21, 11.0))
        self.fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.99)
        self.ax.set(xlim=(0, 225), ylim=(0, 116))
        self.ax.axis('off')
        self.body = FontProperties(fname='C:/Windows/Fonts/msyh.ttc' if self.cn else 'C:/Windows/Fonts/arial.ttf')
        self.bold = FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc' if self.cn else 'C:/Windows/Fonts/arialbd.ttf')
        self.box_checks = []
        self.all_texts = []

    def text(self, x, y, s, size=13, color=None, bold=False, ha='center', va='center', **kw):
        artist = self.ax.text(x, y, s, fontsize=size, fontproperties=self.bold if bold else self.body,
                             color=color or COLORS['ink'], ha=ha, va=va, linespacing=1.35, zorder=5, **kw)
        self.all_texts.append(artist)
        return artist

    def panel(self, x, y, w, h, kind, title, tag, subtitle=None, dashed=False):
        color = COLORS[kind]
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.1',
            facecolor=COLORS[kind+'_fill'], edgecolor=color, linewidth=1.0,
            linestyle=(0,(5,4)) if dashed else '-', zorder=0)
        self.ax.add_patch(patch)
        self.text(x+3, y+h-3.5, f'({tag})  {title}', size=14.5, color=color, bold=True, ha='left')
        if subtitle:
            self.text(x+w-2.5, y+h-3.5, subtitle, size=11.3, color=color, ha='right')

    def node(self, x, y, w, h, title, lines, kind='person', dashed=False, title_size=13.2, body_size=12):
        color = COLORS[kind]
        patch = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.02,rounding_size=.7',
            facecolor=COLORS[kind+'_node'], edgecolor=color, linewidth=1.2,
            linestyle=(0,(4,3)) if dashed else '-', zorder=3)
        self.ax.add_patch(patch)
        tt = self.text(x+w/2, y+h-3.7, title, title_size, color, bold=True)
        self.box_checks.append((tt,(x+.7,y+.6,w-1.4,h-1.2)))
        # Deliberate line positions preserve room for multiline titles.
        top = y+h-8.4 if '\n' not in title else y+h-10.6
        bottom = y+2.4
        ys = [top] if len(lines)==1 else [top+(bottom-top)*i/(len(lines)-1) for i in range(len(lines))]
        for yy,s in zip(ys,lines):
            t = self.text(x+w/2, yy, s, body_size)
            self.box_checks.append((t,(x+.65,y+.55,w-1.3,h-1.1)))

    def arrow(self, points, color=None, dashed=False, width=1.55, arrow=True):
        color = color or COLORS['line']
        codes=[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)
        path=MplPath(points,codes)
        if arrow:
            patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=13.5,
                linewidth=width,color=color,linestyle=(0,(4,3)) if dashed else '-',zorder=2)
        else:
            patch=PathPatch(path,facecolor='none',edgecolor=color,linewidth=width,
                linestyle=(0,(4,3)) if dashed else '-',zorder=2)
        self.ax.add_patch(patch)

    def draw(self):
        cn=self.cn
        self.text(8,112,'旋转分离的第一人称行人轨迹预测' if cn else 'Rotation-Factorized Egocentric Trajectory Forecasting',
                  size=24,bold=True,ha='left')
        self.text(8,106.8,'实际运行架构  |  过去 6 点（0.5 秒） → 未来 15 点（1.5 秒）' if cn else
                  'Implemented architecture  |  6 observed steps (0.5 s)  →  15 future steps (1.5 s)',
                  size=13.5,color=COLORS['muted'],ha='left')

        self.panel(8,78,140,25,'camera','共享相机预测' if cn else 'Shared camera forecast','a')
        self.panel(8,40,140,32,'person','独立行人预测' if cn else 'Independent pedestrian forecast','b')
        self.panel(154,40,67,63,'geometry','几何合成' if cn else 'Geometric composition','c')
        self.panel(8,5,213,28,'training','训练监督与辅助重建' if cn else 'Training supervision and auxiliary reconstruction','d',dashed=True)

        # Camera pathway. Historical Q feeds only deterministic normalization
        # in the pedestrian pathway; predicted Q goes only to reprojection.
        self.node(11,81,25,14,'历史相机旋转' if cn else 'Historical rotations',
                  [r'$R_{t-5:t}$',r'$6\times3\times3$'],kind='camera',body_size=12.5)
        self.node(42,81,26,14,'相对旋转' if cn else 'Relative rotations',
                  [r'$Q_i=R_t^{\mathsf{T}}R_i$',r'$\omega_i=\operatorname{Log}(Q_i)$'],kind='camera')
        self.node(74,81,31,14,'相机预测器' if cn else 'Camera predictor',
                  [r'GRU: $3\rightarrow32$',r'MLP: $32\rightarrow64\rightarrow45$'],kind='camera')
        self.node(116,81,29,14,'未来旋转预测' if cn else 'Future rotations',
                  [r'$\hat{\omega}_{t+1:t+15}\in\mathbb{R}^{15\times3}$',r'$\hat{Q}_k=\operatorname{Exp}(\hat{\omega}_k)$'],kind='camera')
        for left,right in [(36,42),(68,74),(105,116)]:self.arrow([(left,88),(right,88)],COLORS['camera'])
        self.text(110.5,91.2,'冻结' if cn else 'frozen',size=10.5,color=COLORS['camera'])
        self.arrow([(55,81),(55,63.5)],COLORS['camera'])
        self.text(57,75.3,'历史旋转' if cn else 'history only',size=10.5,color=COLORS['camera'],ha='left')

        # Pedestrian pathway.
        self.node(11,45,25,18,'历史行人轨迹' if cn else 'Historical tracks',
                  ['SAM2 掩膜外接框中心' if cn else 'SAM2 mask-box centers',r'$p_{t-5:t}\in\mathbb{R}^{6\times2}$','有效性与时间' if cn else 'validity + timestamps'],body_size=11.4)
        self.node(42,45,26,18,'历史旋转补偿' if cn else 'Rotation compensation',
                  [r'$\bar p_i=\pi(KQ_iK^{-1}\tilde p_i)$','归一化中心与位移' if cn else 'normalized xy + displacement',r'$[u_i,\ \Delta u_i,\ v_i,\ \tau_i]\in\mathbb{R}^{6}$'],body_size=11.1)
        self.node(74,45,31,18,'行人编码器' if cn else 'Pedestrian encoder',
                  [r'GRU: $6\rightarrow64$',r'$[h; o]\in\mathbb{R}^{66}$',r'MLP: $66\rightarrow95\rightarrow64$'],body_size=12.0)
        self.node(116,45,29,18,'未来 flow 解码器' if cn else 'Future flow decoder',
                  [r'$15\times2$ '+('归一化点' if cn else 'normalized points'),
                   '5 个起点 · 10 步 · 均值' if cn else '5 starts · 10 steps · mean',
                   '恢复像素尺度与原点' if cn else 'restore pixel scale + origin'],body_size=11.1)
        for left,right in [(36,42),(68,74),(105,116)]:self.arrow([(left,54),(right,54)],COLORS['person'])
        self.text(110.5,57.4,r'$h_p$',size=12.5,color=COLORS['person'])
        self.text(80,42.4,r'$o=p_t/(W,H)$',size=10.5,color=COLORS['muted'])
        self.text(12,65.9,'每人独立输入；共享模型参数' if cn else 'Per-person inputs; shared predictor weights',size=10.7,color=COLORS['person'],ha='left')

        # Two streams join at a fixed geometric operator.
        self.arrow([(145,88),(171,88),(171,64)],COLORS['camera'],width=1.8)
        self.text(174,78,'所有行人共用' if cn else 'Shared across\npedestrians',size=12,color=COLORS['camera'],ha='left')
        self.text(168,73.2,r'$\hat Q_{t+1:t+15}$',size=14,color=COLORS['camera'],ha='right')
        self.node(158,45,27,19,'固定重投影' if cn else 'Fixed reprojection',
                  [r'$\hat H_k=K\hat Q_k^{\mathsf{T}}K^{-1}$',r'$\hat p_k=\pi(\hat H_k\tilde{\hat{\bar p}}_k)$',
                   '无学习参数' if cn else 'No learned parameters'],kind='geometry',body_size=11.8)
        self.arrow([(145,54),(158,54)],COLORS['person'],width=1.8)
        self.text(151.4,58,r'$\hat{\bar p}_{t+1:t+15}$',size=12,color=COLORS['person'])
        self.node(191,45,27,19,'图像轨迹输出' if cn else 'Image-space forecast',
                  [r'$\hat p_{t+1:t+15}$',r'$15\times2$','未来 1.5 秒' if cn else 'next 1.5 s'],kind='geometry',body_size=12.4)
        self.arrow([(185,54),(191,54)],COLORS['geometry'],width=1.8)

        # History head shares only the pedestrian encoder/context. Dashed
        # pathways are training-time use; no historical reconstruction required
        # in future inference.
        self.node(74,9,31,15,'历史 flow 解码器' if cn else 'History flow decoder',
                  [r'$6\times2$ '+('历史重建' if cn else 'reconstruction'),'辅助训练头' if cn else 'auxiliary training head'],
                  kind='training',dashed=True,body_size=12)
        self.arrow([(89.5,45),(89.5,24)],COLORS['training'],dashed=True)
        self.text(92,36.8,r'$h_p$',size=12.5,color=COLORS['training'],ha='left')
        self.node(116,9,61,15,'行人分支训练目标' if cn else 'Pedestrian training objective',
                  [r'$\mathcal{L}_{\mathrm{ped}}=\mathcal{L}_{\mathrm{future}}+0.5\,\mathcal{L}_{\mathrm{history}}$',
                   '归一化坐标中的 flow 数据预测 MSE' if cn else 'Flow data-prediction MSE in normalized coordinates'],
                  kind='training',dashed=True,body_size=12)
        self.arrow([(105,16.5),(116,16.5)],COLORS['training'],dashed=True)
        self.arrow([(130.5,45),(130.5,36.5),(146.5,36.5),(146.5,24)],COLORS['training'],dashed=True)
        self.node(187,9,31,15,'参考轨迹目标' if cn else 'Reference targets',
                  ['历史与未来旋转归一化轨迹' if cn else 'rotation-normalized tracks',
                   '训练监督使用' if cn else 'history + future; supervision'],kind='training',dashed=True,body_size=11.2)
        self.arrow([(187,16.5),(177,16.5)],COLORS['training'],dashed=True)
        self.text(12,22.2,'相机分支单独预训练' if cn else 'Camera branch: separate pretraining',size=12.4,color=COLORS['training'],bold=True,ha='left')
        self.text(12,16.8,r'$\mathcal{L}_{\mathrm{cam}}=\operatorname{MSE}(\hat\omega,\omega^{\mathrm{ref}})$',size=13,ha='left')
        self.text(12,11,'行人训练时保持相机权重冻结' if cn else 'Camera weights stay frozen in pedestrian training',size=11.5,color=COLORS['muted'],ha='left')

        # Legend / scope definition: precise about the implemented decomposition.
        self.text(156,94,'两个预测分支仅在此合成' if cn else 'Forecast streams meet only here',size=11.1,color=COLORS['geometry'],ha='left')
        self.arrow([(159,42.4),(165,42.4)],COLORS['line'],width=1.3)
        self.text(167,42.4,'推理路径' if cn else 'inference',size=10.4,ha='left',color=COLORS['muted'])
        self.arrow([(192,42.4),(198,42.4)],COLORS['training'],dashed=True,width=1.3)
        self.text(200,42.4,'训练' if cn else 'training',size=10.4,ha='left',color=COLORS['muted'])
        self.text(8,2.0,'注：仅补偿旋转；剩余轨迹仍包含平移、视差、行人运动和噪声。' if cn else
                  'Scope: rotation compensation only; residual tracks retain translation, parallax, pedestrian motion and noise.',
                  size=10.5,color=COLORS['muted'],ha='left')
        self.text(143,2.0,r'$u_i=\bar p_i/(W,H)-o$; '+('K：相机内参' if cn else 'K: camera intrinsics'),
                  size=10.5,color=COLORS['muted'],ha='left')

    def validate(self):
        """Check text extents against node bounds and canvas; tighten long labels."""
        adjustments=[]
        for attempt in range(12):
            self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();offenders=[]
            for artist,(x,y,w,h) in self.box_checks:
                bb=artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
                if bb.x0<x or bb.x1>x+w:
                    offenders.append(artist)
            if not offenders:break
            for artist in offenders:
                old=artist.get_fontsize();artist.set_fontsize(old-.35)
                adjustments.append(dict(text=artist.get_text(),old=old,new=old-.35))
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
        violations=[]
        for artist,(x,y,w,h) in self.box_checks:
            bb=artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            if bb.x0<x-.01 or bb.x1>x+w+.01 or bb.y0<y-.01 or bb.y1>y+h+.01:
                violations.append(dict(text=artist.get_text(),bounds=[bb.x0,bb.y0,bb.x1,bb.y1],node=[x,y,w,h]))
        for artist in self.all_texts:
            bb=artist.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            if bb.x0<0 or bb.x1>1 or bb.y0<0 or bb.y1>1:violations.append(dict(text=artist.get_text(),outside_canvas=True))
        if violations:raise AssertionError(json.dumps(violations,ensure_ascii=False,indent=2))
        return dict(node_text_count=len(self.box_checks),canvas_text_count=len(self.all_texts),font_adjustments=adjustments,text_bounds_passed=True)

    def save(self):
        self.draw();qa=self.validate();base=DEST/f'architecture_{self.language}'
        for ext in ['png','pdf','svg']:
            self.fig.savefig(base.with_suffix('.'+ext),dpi=240,facecolor='white',transparent=False,
                             metadata={'Creator':'Matplotlib; implementation-based architecture diagram'} if ext in ['pdf','svg'] else None)
        self.fig.savefig(DEST/f'architecture_{self.language}_preview.png',dpi=120,facecolor='white')
        plt.close(self.fig)
        return qa


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    fontManager.addfont('C:/Windows/Fonts/arial.ttf')
    plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.unicode_minus':False,'savefig.pad_inches':.1})
    qa={lang:Architecture(lang).save() for lang in ['en','zh']}
    sources=[ROOT/'scripts/factorized_forecast_v1_20261002'/f for f in ['network.py','shared.py','run.py']]
    sources += [ROOT/'scripts/biflow_aria_v1_20261002/model.py',ROOT/'outputs/factorized_forecast_v1_20261002/protocol.json']
    record=dict(renderer='Matplotlib, code-generated vector graphics; no image generation',scope='Implemented separated_dual rotation-only model',
                renderer_script=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=digest(Path(__file__))),
                sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources},layout_qa=qa,
                files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in DEST.iterdir() if p.suffix in ['.png','.pdf','.svg']})
    (DEST/'diagram_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'directory':str(DEST),'files':list(record['files']),'bounds_passed':True},ensure_ascii=False))


if __name__=='__main__':main()
