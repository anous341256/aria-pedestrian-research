"""Freeze the observational near-orthogonal design and render its audit."""
from design import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

NAMES=['累计旋转','净旋转','相机平移速度','人物框高比例','尺度变化率','补偿后横向速度','补偿后纵向速度','非匀速残差']
UNITS=['度','度','米/秒','框高/图高','log框高/秒','图宽/秒','图高/秒','残差/框高']

def table(head,rows):
 return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
 rows=load();audit=read(OUT/'design_audit.json');wr=read(OUT/'analysis_weights.json');n=len(rows)
 assert [r['key'] for r in rows]==[r['key'] for r in wr]
 w=np.array([r['weight'] for r in wr]);x=np.array([r['levels'] for r in rows],float)
 recs=sorted({r['recording'] for r in rows});rc=np.array([[r['recording']==rec for rec in recs] for r in rows],float)
 gram=np.array(audit['weighted_main_gram']);raw=np.corrcoef(x.T);adjusted=np.array(audit['recording_adjusted_correlations'])
 triple=np.array([np.prod(x[:,cols],axis=1) for cols in itertools.combinations(range(8),3)])
 # Audit all nuisance-adjusted leave-one-recording designs before any outcomes.
 loo=[]
 for rec in recs:
  keep=np.array([r['recording']!=rec for r in rows]);a=np.c_[rc[keep][:,np.any(rc[keep],axis=0)],x[keep]]
  loo.append(dict(excluded=rec,rows=int(keep.sum()),rank=int(np.linalg.matrix_rank(np.sqrt(w[keep,None])*a)),columns=a.shape[1]))
 assert all(r['rank']==r['columns'] for r in loo)
 unstable=np.array([abs(r['features']['H']-read(SOURCE/'thresholds_locked.json')['thresholds']['H'])<=r['sensitivity']['H'] for r in rows])
 # Recording-adjusted alias matrix of omitted two-factor terms into main effects.
 z=rc;rx=x-z@np.linalg.solve(z.T@(w[:,None]*z),z.T@(w[:,None]*x))
 interactions=np.array([x[:,i]*x[:,j] for i,j in itertools.combinations(range(8),2)]).T
 adjusted_alias=np.linalg.solve(rx.T@(w[:,None]*rx),rx.T@(w[:,None]*interactions))
 extra=dict(unweighted_max_abs_factor_correlation=float(abs(raw-np.eye(8)).max()),weighted_max_abs_factor_correlation=float(abs(gram-np.eye(8)).max()),recording_adjusted_max_abs_factor_correlation=float(abs(adjusted-np.eye(8)).max()),full_design_rank=int(np.linalg.matrix_rank(np.c_[rc,x])),full_design_columns=17,leave_one_recording=loo,H_sensitivity_flagged=int(unstable.sum()),H_flagged_weight=float(w[unstable].sum()),max_adjusted_omitted_interaction_alias=float(abs(adjusted_alias).max()),main_by_interaction_alias=adjusted_alias.tolist(),interaction_names=[''.join(F[k] for k in pair) for pair in itertools.combinations(range(8),2)])
 write(OUT/'robustness_design_audit.json',extra)
 # Keep all naturally present patterns; these are descriptive strata, not 16 runs.
 patterns=[]
 for level in sorted(set(map(tuple,x.astype(int)))):
  ii=np.array([tuple(r['levels'])==level for r in rows]);rr=[r for r,b in zip(rows,ii) if b]
  patterns.append(dict(levels=list(map(int,level)),windows=len(rr),weight=float(w[ii].sum()),recordings=sorted({r['recording'] for r in rr}),keys=[r['key'] for r in rr]))
 write(OUT/'observed_patterns.json',patterns)
 np.savez_compressed(OUT/'design_matrix.npz',levels=x,weights=w,recording_dummies=rc)
 # Preserve the established no-overlap lists as diagnostics, without rebalancing
 # on error or presenting these subsets as the same exact weighted design.
 import sys
 sys.path.insert(0,str(ROOT/'scripts/g1_factor_inventory_20261006'))
 from inventory import pack_nonoverlap
 write(OUT/'sensitivity_subsets.json',dict(person_nonoverlap=[r['key'] for r in pack_nonoverlap(rows)],recording_time_nonoverlap=[r['key'] for r in pack_nonoverlap(rows,False)],H_stable=[r['key'] for r,b in zip(rows,unstable) if not b]))
 figures=OUT/'figures';figures.mkdir(exist_ok=True)
 plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
 plt.rcParams['axes.unicode_minus']=False
 fig,axes=plt.subplots(1,3,figsize=(12.5,4.6))
 for a,c,title in zip(axes,[raw,gram,adjusted],['原始等权样本','新版加权设计','控制录制差异后']):
  a.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
  a.set(xticks=range(8),xticklabels=F,yticks=range(8),yticklabels=F,title=title)
  for i in range(8):
   for j in range(8):a.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(c[i,j])>.6 else '#152330')
 fig.suptitle('8因素对照的相关性检查 · 同一批214个窗口',fontsize=14)
 fig.text(.04,.04,'阈值和轨迹不变；调整统计权重。中图两两相关绝对值≤0.05。\n仅针对二水平主效应编码；不代表物理独立、连续值解耦或交互作用消失。',fontsize=10)
 fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(figures/'design_balance.png',dpi=180);plt.close(fig)
 threshold=read(SOURCE/'thresholds_locked.json')['thresholds']
 ftable=table(['因素','含义及单位','固定阈值','原始低/高条数','加权低/高占比'],[[F[j],f'{NAMES[j]}（{UNITS[j]}）',f'{threshold[F[j]]:.6g}',f"{audit['factor_levels'][F[j]]['low_count']} / {audit['factor_levels'][F[j]]['high_count']}",'50% / 50%'] for j in range(8)])
 rectable=table(['录制尾号','原始窗口数','加权占比'],[[rec.split('_')[-1],sum(r['recording']==rec for r in rows),f"{audit['recording_weights'][rec]:.2%}"] for rec in recs])
 plan=f'''# G1现有样本的八因素加权近似正交对照

使用已有214个可评价窗口，保留8个候选因素，通过受约束的统计权重建立近似正交的主效应筛选。每个因素加权后低高各占50%，任意两因素相关系数绝对值不超过0.05。采用全部现有可评价样本，不再等待原L16空格补齐。

## 研究问题与数据

问题仍是固定预测模型后，哪些历史可见因素与未来1.5秒预测误差和相对保持最后位置的收益有关。不是比较训练子集策略，也不进行16次重新训练。

总候选268个中，214个具有最近`matched97_new`权重库的匹配录制留出资格，覆盖9个录制。相机与行人网络均未用对应测试录制进行训练或验证选模。214个全部使用，包含{len(patterns)}种实际出现的八因素高低组合；其余54个保留作特征分布资料，不使用看过该录制的模型评分。六架构和三个训练种子全部保留，不混用另一个权重库。

每条观测是6个历史点加15个未来点。二水平因素只用过去0.5秒的6点计算，沿用补标前已冻结的阈值。全未来目标只用于计算响应；新设计不读取ADE/FDE挑样本、调阈值或权重。

{ftable}

F/G高低表示有符号速度的相对水平，不是绝对运动幅度。E包含放大和缩小方向。C来自Aria参考相机中心运动；旋转补偿残差仍含平移、视差和标注噪声，不是纯真实行人运动。

## 如何构成对照

原L16要求找到16行指定组合。新版从已有全部真实组合出发，让每条窗口获得一个统计权重。常见组合的总占比降低、稀少组合适度提高，使比较某因素高低时，其他单个因素的分布尽量接近。

1. 所有214条都有正权重；每条限制在等权值的0.25到4倍。不会复制样本、修改轨迹或拼造组合。
2. 所有权重之和为1，每个因素的加权均值为0，即低高各50%。
3. 每个录制占比限制在2%到25%，避免被一个大录制主导；不是声称九个录制等权。
4. 在这些约束下，先最小化28对因素中最大的绝对相关。本数据能达到约{audit['minimax_abs_pair_correlation']:.4f}；按预定0.025步长向上取允许界限0.05。
5. 在相关≤0.05范围内，最小化权重平方和，使权重尽可能均匀。最终加权两因素的低低、低高、高低、高高四种组合各占23.75%至26.25%。

这叫**受约束加权近似正交设计**。它不是把自然样本变成随机试验，也不是D最优设计；优化目标已经明确为两两相关及权重均匀度。

## 已经核验的设计性质

- 加权前二水平因素最大绝对相关为{extra['unweighted_max_abs_factor_correlation']:.3f}；加权后≤0.050。该结论限于高低编码，连续变量并未被强行解耦。
- 每条实际权重为等权的{audit['actual_relative_weight_range'][0]:.2f}至{audit['actual_relative_weight_range'][1]:.2f}倍。仅按权重离散度计算的Kish有效样本量约{audit['ess_weights_only']:.1f}；未扣除重叠和录制相关性，不是约105条独立样本。
- 控制9个录制截距后，8个因素的设计矩阵仍满秩；条件数{audit['recording_adjusted_condition']:.2f}，最大VIF为{max(audit['recording_adjusted_vif']):.2f}，当前不存在严重线性共线性。每次删去一个录制后也保持满秩。
- 控制录制后因素最大绝对相关约{extra['recording_adjusted_max_abs_factor_correlation']:.3f}，因此正式效应使用联合回归，不将高低误差直接相减视为完全独立效应。

![设计相关性核验](../outputs/g1_factor_redesign_20261006/figures/design_balance.png)

{rectable}

## 固定模型的评价流程

对同一批214条窗口，分别调用原六架构的匹配留出权重，保持历史输入、6点/15点、坐标变换和预测时长不变。每架构每个窗口对3个种子的评价值取平均后作主分析，逐种子另作稳定性检查；种子不增加数据样本量。

主响应是图像空间ADE（像素）。辅助响应为FDE、ADE除以历史末帧框高，以及“保持最后位置ADE减模型ADE”。最后一项大于0表示模型优于保持最后位置。

每个架构、每个响应分别拟合加权模型：

`响应 = 录制截距 + βA·A + βB·B + … + βH·H + 残差`。

A–H编码为−1/+1，报告2β，作为在此加性模型下、控制其他七因素和录制后，从低水平到高水平的调整差。它是当前G1与固定模型的关联，不是某物理因素的因果效应。ADE差为正表示高水平更难；收益差为正表示该模型在高水平下相对保持最后位置更有价值。

同时展示加权高低均值和实际连续值散点，帮助理解方向。连续值趋势只作伴随诊断，不能将连续模型也称为正交。

## 重叠 交互与稳定性

214条来自9个录制，包含重叠窗口和同步人物。推断以录制为聚类单位，保留同录制内所有相关性；报告留一录制的效应范围、符号稳定性，并以录制聚类重采样给出探索性区间。9个录制较少，不以普通独立样本ANOVA或单个显著性值确定因素排名。

主结论并列以下敏感性结果：不加设计权重；92个人内不重叠窗口；52个录制时间不重叠窗口；删除H敏感窗口；去掉H的七因素联合模型。这些子集沿用固定权重后归一化，不强称它们仍达到原来的50/50和0.05，需同步报告其平衡性。

H的删除点敏感性标记涉及{extra['H_sensitivity_flagged']}/{n}个窗口，占新设计权重{extra['H_flagged_weight']:.1%}；它可能同时反映运动非匀速和mask噪声。H结果须附上述检查。

新版仅让主效应之间近似正交，**不具有旧L16的分辨率IV保证**。现有支持无法同时令所有主效应与二因素交互正交；记录调整后，潜在遗漏交互进入主效应估计的最大线性系数约{extra['max_adjusted_omitted_interaction_alias']:.3f}。因此首轮重点筛主效应，不一次拟合28个交互或声称已排除交互。A×F、A×G、C×H、E×G作为预先列出的敏感性项，逐项加入联合模型观察主效应是否变化，不能凭其中一次较好就挑选结论。若方向随交互设定明显变化，应标记为耦合关联待查。

## 预定输出

1. 八因素×六架构的调整后ADE差及留一录制范围。
2. 同口径的保持最后位置相对收益差，区分“场景本身运动更大”与“模型更擅长或更不擅长该场景”。
3. 不同种子、非重叠子集和H测量敏感性对方向的影响。
4. 因素分为当前证据较一致、依赖模型或录制、数据暂不支持判断；不强制给出一个全局重要性榜单。

## 版本与依据

本版已固定具体214个窗口和每条权重，并完成设计矩阵检查；尚未运行预测误差分析。原L16、旧阈值、数据、模型及原盘点报告保留。本版替代继续补齐16格的计划，不新增标注或训练。

约束候选集合需要计算机辅助设计；[NIST计算机辅助最优设计说明](https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.htm)解释了为什么受限空间下经典设计可能不适用，以及设计最优性依赖指定模型。本版使用上述自定义平衡目标，未将其称作D最优。主效应与交互混杂的解释见[NIST Confounding](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm)。

具体清单与核验：`outputs/g1_factor_redesign_20261006/analysis_weights.json`、`design_audit.json`、`robustness_design_audit.json`、`observed_patterns.json`及`design_matrix.npz`。全部选择基于冻结的历史特征及录制信息。
'''
 (ROOT/'plans/G1_FACTOR_SCREENING_REDESIGN_20261006.md').write_text(plan,encoding='utf8')
 protocol=dict(version='g1_factor_redesign_20261006',task='fixed-model observational scenario screening',model_bank='matched97_new',factors=F,thresholds=threshold,past_points=6,future_points=15,dt_seconds=.1,primary_response='image_ADE_pixels',secondary_responses=['image_FDE_pixels','ADE_over_last_history_bbox_height','hold_ADE_minus_model_ADE'],factor_coding=[-1,1],analysis='separate architecture WLS; 9 recording intercepts and 8 main effects; report twice each coefficient',seed_aggregation='mean response per window across seeds 17,29,43; seedwise sensitivity only',inference='recording-cluster exploratory intervals and leave-one-recording ranges; small-cluster limitation',weight_rule='bounded minimax plus minimum sum squares',all_windows_used=214,annotation_or_training=False,prediction_scores_read=False,interactions_sensitivity=['AF','AG','CH','EG'],main_effect_resolution_IV=False)
 write(OUT/'protocol.json',protocol)
 for p,h in audit['protected_files'].items():assert digest(ROOT/p)==h,p
 files=['analysis_weights.json','design_audit.json','robustness_design_audit.json','observed_patterns.json','design_matrix.npz','sensitivity_subsets.json','protocol.json']
 write(OUT/'manifest.json',dict(files={f:digest(OUT/f) for f in files},source_files=audit['protected_files'],all_source_hashes_unchanged=True))
 print('Saved design:',n,'windows;',len(patterns),'patterns; adjusted rank',extra['full_design_rank'],'; max correlation',extra['weighted_max_abs_factor_correlation'],flush=True)

if __name__=='__main__':main()
