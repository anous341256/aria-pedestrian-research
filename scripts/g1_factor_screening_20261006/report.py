"""Research figures and a full report of all predeclared factor comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from setup import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FACTORS=list('ABCDEFGH')
FNAMES=['累计旋转','净旋转','相机平移速度','人物框高','尺度增长率','横向有符号速度','纵向有符号速度','非匀速残差']
MNAMES=['先融合','仅行人','独立分离','双向交互','单向交互','交互后重投影']
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def rng(a):return f'{np.min(a):+.1f}～{np.max(a):+.1f}'

def main():
 assert read(OUT/'verification.json')['passed']
 dest=ROOT/'reports/g1_factor_screening_20261006';dest.mkdir(parents=True,exist_ok=True)
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 z=np.load(OUT/'analysis_arrays.npz');e=z['effects'];overall=read(OUT/'overall.json');results=read(OUT/'factor_effects.json');sens=read(OUT/'sensitivity.json');va=read(OUT/'verification.json')
 plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
 plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(13,6.3))
 for ax,k,title,lim,cmap in [(axes[0],0,'ADE变化：红色表示高水平更难',40,'RdBu_r'),(axes[1],3,'相对保持最后位置的收益变化：蓝色更有利',25,'RdBu')]:
  ax.imshow(e[:,:,k],vmin=-lim,vmax=lim,cmap=cmap,aspect='auto')
  ax.set(xticks=range(6),xticklabels=['A\n先融合','B\n仅行人','C\n独立分离','D\n双向交互','E\n单向交互','F\n交互重投影'],yticks=range(8),yticklabels=[f'{f} {s}' for f,s in zip(FACTORS,FNAMES)],title=title)
  for i in range(8):
   for j in range(6):ax.text(j,i,f'{e[i,j,k]:+.1f}',ha='center',va='center',color='white' if abs(e[i,j,k])>.65*lim else '#162b35',fontsize=10)
 fig.suptitle('G1八因素筛选 · 高水平减低水平的调整差（像素）',fontsize=15)
 fig.text(.04,.028,'214个窗口 / 9个录制；固定模型，3种子平均；控制其他7因素及录制差异。\n表内为点估计，不代表均已排除零效应。F/G保留运动方向；重复窗口与种子不是独立样本。',fontsize=10)
 fig.tight_layout(rect=[0,.10,1,.95]);fig.savefig(figdir/'factor_effects.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10.5,4.8));methods=['hold_last']+VARIANTS;labels=['保持最后位置']+MNAMES;xx=np.arange(7)
 ax.bar(xx-.18,[overall[v]['unweighted']['ADE'] for v in methods],.36,color='#a2adb7',label='原始等权')
 bars=ax.bar(xx+.18,[overall[v]['weighted']['ADE'] for v in methods],.36,color='#2e7987',label='设计权重')
 for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}',ha='center',fontsize=10)
 ax.set(xticks=xx,xticklabels=labels,ylabel='未来1.5秒ADE（像素，越低越好）',ylim=(0,165),title='同一批214个G1窗口的整体误差');ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.text(.08,.018,'权重改变用于分析的场景占比；加权与等权的数值差不是模型训练带来的改善。',fontsize=10)
 fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(figdir/'overall_ADE.png',dpi=180);plt.close(fig)
 # Show actual leave-one-recording sensitivity for every factor and architecture.
 loo=z['loo_effects'][:,:,:,0]
 fig,axes=plt.subplots(2,4,figsize=(14,6.5),sharex=True)
 colors=['#248593','#526a84','#d4843b','#59945f','#886b9b','#be6060']
 for f,ax in enumerate(axes.flat):
  for j,c in enumerate(colors):
   ax.plot([loo[:,f,j].min(),loo[:,f,j].max()],[j,j],color=c,lw=2)
   ax.scatter(e[f,j,0],j,color=c,s=22,zorder=3)
  ax.axvline(0,color='#888',lw=.8,ls='--');ax.set(title=f'{FACTORS[f]} {FNAMES[f]}',yticks=range(6),yticklabels=list('ABCDEF'));ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
 for ax in axes[1]:ax.set_xlabel('高减低ADE差（像素）')
 fig.suptitle('逐个去掉一个录制后，因素方向是否保持？',fontsize=14)
 fig.text(.04,.014,'点：完整设计估计；线：9次留一录制估计范围。线段不是置信区间。模型A–F与主图一致。',fontsize=10)
 fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(figdir/'leave_one_recording.png',dpi=180);plt.close(fig)
 main=table(['方法','等权ADE','设计加权ADE','设计加权FDE','加权相对保持收益'],[[name,f"{overall[v]['unweighted']['ADE']:.2f}",f"{overall[v]['weighted']['ADE']:.2f}",f"{overall[v]['weighted']['FDE']:.2f}",f"{overall[v]['weighted']['gain_ADE']:+.2f}"] for v,name in zip(methods,labels)])
 factorrows=[]
 notes=[
  '主分析均为正，但录制时间不重叠子集中部分直接输出模型反向；证据不稳定。',
  '模型间方向不同，留一录制可翻转；当前不能得出稳定影响。',
  '独立分离/交互重投影增加约21～22像素，其余为−0.6～+5.9；呈现模型依赖，区间仍宽。',
  '像素误差较大，但除以框高后方向反转；应保留为尺度控制，不能等同于近处更难。',
  '六模型、三个种子、留一录制及预设敏感性检查均保持正方向；本轮最一致的难度线索。',
  '有符号方向，不是运动绝对幅度；部分直接输出模型在子集分析反向。',
  '四个直接输出模型负方向较稳定；两种重投影模型可翻转，需结合尺度增长联合看。',
  '主分析均为正，但改用等权后六模型均反向；对场景权重敏感，尚难拆清运动弯曲与标注抖动。']
 for f in range(8):factorrows.append([f'{FACTORS[f]} {FNAMES[f]}',rng(e[f,:,0]),rng(e[f,:,3]),notes[f]])
 ft=table(['因素','六模型调整ADE差范围','相对保持收益差范围','解释'],factorrows)
 detail=[]
 for r in results:
  d=r['metrics']['ADE'];m=VARIANTS.index(r['variant']);detail.append([r['factor'],MNAMES[m],f"{d['effect']:+.2f}",f"[{d['bootstrap95'][0]:+.2f}, {d['bootstrap95'][1]:+.2f}]",f"[{d['loo_range'][0]:+.2f}, {d['loo_range'][1]:+.2f}]",f"{d['loo_same_sign']}/9",f"{d['seed_same_sign']}/3"])
 appendix='# 八因素主效应明细\n\n每项为调整后的高水平减低水平ADE差。2000次录制聚类重采样百分位区间仅作探索，未作多重比较显著性判定。\n\n'+table(['因素','模型','ADE差 px','探索性95%区间','留一录制范围','留一方向一致','种子方向一致'],detail)+'\n'
 (dest/'EFFECTS.md').write_text(appendix,encoding='utf8')
 st=[]
 for f in range(8):
  st.append([f'{FACTORS[f]} {FNAMES[f]}']+[rng(np.array(sens[k]['effect'],float)[f,:,0]) if f!=7 or k!='without_H' else '未拟合H' for k in ['unweighted','person_nonoverlap','recording_time_nonoverlap','H_stable','without_H']])
 stable=table(['因素','等权214','人内不重叠92','录制时间不重叠52','H稳定142','去掉H'],st)
 report=f'''# G1八因素固定模型筛选结果

日期：2026-10-06。已完成预定214个窗口、9个录制、六种架构、三个种子的固定模型评价，共3,852条窗口预测。全部采用历史6点预测未来15点，时长1.5秒。无重新训练、阈值重选或按结果调整统计权重。

## 首轮结论

**当前最一致的场景难度线索是历史人物框尺度增长率E。** 在控制其他七因素及录制后，高水平组的ADE在六模型中增加约28～32像素；三个种子、逐个去掉录制，以及全部预设敏感性比较的方向均为正。不过效应大小随子集变化，不能把28～32像素当作普适常数。

保持最后位置在E上的相同调整差为 **{z['hold_effects'][4,0]:.1f}像素**。六模型相对保持最后位置的收益差只有 **{rng(e[4,:,3])}像素**。所以E目前更像共同场景难度的描述，不是已证明某模型专门利用了这个因素，更不是已证明按E增加训练样本会提升精度。

人物框高D是必要的尺度控制；补偿后的纵向有符号速度G在四种直接输出模型中方向较稳定。相机平移C对当前两种带重投影系统的点估计影响更大。A/B相机旋转、F横向方向和H非匀速残差目前存在明显模型或子集依赖。

## 同批样本整体误差

{main}

收益定义为“保持最后位置ADE减模型ADE”，正值才表示模型更好。这一批上六模型的加权ADE都高于保持最后位置；仅行人模型与保持最后位置只差约0.85像素，不据此宣称稳定优劣。部分模型FDE稍低于保持最后位置，但平均路径ADE没有改善。

这些数值对应本轮严格G1子集和特定分析权重，不能与原有不同样本、不同加权方式的ADE直接当作训练前后提升比较。

![同批整体ADE](../../outputs/g1_factor_screening_20261006/figures/overall_ADE.png)

## 八个因素的对照结果

{ft}

所有差值均为“高减低”，来源于每个模型的联合加权回归：录制截距+八因素。正ADE差表示高水平在当前加性调整模型下误差更大。它们是观察关联，不是随机操纵因素得出的因果贡献。

E的高水平指过去0.5秒log框高斜率>0.1131/秒，即画面中的人物尺度增长较快；不是尺度变化绝对值较大，也不能直接叫真实接近速度。F/G保留方向；例如G较高不能解释成“竖直运动越大越容易”。

![八因素效应](../../outputs/g1_factor_screening_20261006/figures/factor_effects.png)

D的像素ADE调整差为+13.7～+17.5像素，但ADE除以历史末帧框高后的差为−0.193～−0.175个框高。方向反转说明尺度判断依赖误差单位，并且归一化分母本身与D有关，不能据此直接断言近处更容易或更困难。

C在独立分离和交互后重投影模型中，ADE差分别为+20.9、+22.5像素；相对保持最后位置收益差为−20.0、−21.6像素。该方向在这两模型的留一录制和种子检查中保持，但聚类区间跨零。它值得作为模型特定条件继续分析，尚不能归因为某一种几何机制或“分解必然有害”。

## 稳定性与不确定性

![留一录制范围](../../outputs/g1_factor_screening_20261006/figures/leave_one_recording.png)

主分析对三个种子的响应先平均；种子不是新增独立数据。2000次以录制为单位的成组重采样全部可估计。仅有9个录制，区间是探索性的；没有把48个因素/模型组合中区间不跨零的项当作经过多重比较确认的发现。

E在六模型中的留一录制和种子方向均保持正。其探索性95%区间在五个模型中不跨零，独立分离模型为约[−0.3,52.7]像素。G在四个直接输出模型中负方向较一致，两种重投影模型区间更宽且可翻转。其余大多数因素区间跨零。

以下每格是六模型的ADE差范围，使用预先确定的样本和固定权重：

{stable}

H稳定子集有142个窗口、8个录制；并非原214条的同分布重复。去掉72个H敏感窗口后，H效应仍为正但幅度减小；改用原始等权时，H在六模型中都由正变负。录制时间不重叠子集还使部分旋转和横向速度效应反转。这说明结果依赖分析权重及样本构成，不能单独据此证明标注噪声是原因。

子集没有重新平衡权重，其相关性与样本量不同，具体数值在`sensitivity.json`。四个预定交互AF、AG、CH、EG分别加入联合模型的结果也已保留；近似正交只针对主效应编码，没有恢复原L16的分辨率IV性质。全48项效应、探索区间与方向检查见[EFFECTS.md](EFFECTS.md)。

## 对后续分布描述的建议

第一优先保留 **E尺度增长率** 描述难度变化，保留 **D表观尺度** 控制像素量纲；同时保留 **G有符号纵向运动** 并与E联合看。**C相机平移**应按模型家族单列，而不是用一个跨模型平均值掩盖差异。A/B/F/H继续记录，当前不宜依据本轮给它们设定确定的采样重要性权重。

本轮没有训练集抽样干预，因此这些建议用于下一步分布描述和检验假设，尚未证明任何新抽样方案可以提高训练精度。

## 数据与验证

- 历史特征、低高阈值和逐窗口权重均与已保存设计完全一致。214条来自112个实际高低组合，每条权重为等权0.25～4倍；加权二水平因素最大相关≤0.05。
- 本轮使用最近matched97_new的录制留出权重：162个人物预测模型、27个相机模型。相机和人物两侧训练/验证均排除对应测试录制。
- 原始批次重放全部通过，最大预测差 **{va['original_replay_max_px']:.1f}像素**。物理移除未来数组后，当前预测最大变化 **{va['future_removed_max_px']:.1f}像素**。
- 没有无效投影罚分：六模型的无效预测步数均为0。独立重算ADE完全一致；使用SVD最小二乘复算全部主效应，与原实现最大差小于1e−10像素。
- 历史补偿按过去R/K独立重算，最大坐标差约{va['history_geometry_max_px']:.6f}像素；226个受保护数据/设计/代码/权重文件哈希一致。
- 标签为离线SAM2与代理抽样核查参考，历史姿态来自Aria离线MPS。它不是独立人工GT或RGB-only在线结果。旧留出数据被多次查看，所有结论属于开发阶段证据。

主要文件：`outputs/g1_factor_screening_20261006/`下的预测数组、逐窗口误差、整体统计、因素效应、敏感性检查、聚类重采样与验证记录。脚本位于同名版本目录；旧数据与模型保留。
'''
 (dest/'RESULTS.md').write_text(report,encoding='utf8')
 write(OUT/'report_assets.json',dict(report=str(dest/'RESULTS.md'),appendix=str(dest/'EFFECTS.md'),figures=[str(p) for p in figdir.glob('*.png')]))
 print('Report saved',dest/'RESULTS.md',flush=True)

if __name__=='__main__':main()
