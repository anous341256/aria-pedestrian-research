"""Write verified measurements plus separately reviewed interpretations."""
from shared import *
from datetime import datetime

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
 s=read(OUT/'summary.json');ver=read(OUT/'verification.json');geo=read(OUT/'geometry_audit.json');cam=read(OUT/'camera_audit.json');media=read(OUT/'presentation/video_checks.json');findings=read(OUT/'findings.json')
 assert all(a['passed'] for a in [ver,geo,cam,media]);REPORT.mkdir(exist_ok=True,parents=True)
 cfg=read(OUT/'protocol.json');meta=read(DATA/'samples.json');sel=read(DATA/'selection.json')
 methods=VARIANTS+['hold_last','raw_cv'];fmt=lambda z:f'{z:.2f}'
 mainrows=[]
 for v in methods:
  a=s['reviewed'][v];mainrows.append([NAMES.get(v,v),fmt(a['window_mean']['ADE15']),fmt(a['window_mean']['FDE15']),fmt(a['recording_equal']['ADE15']),fmt(s['sparse'][v]['window_mean']['ADE15'])])
 cohortrows=[]
 for v in VARIANTS+['hold_last']:
  cohortrows.append([NAMES.get(v,v)]+[fmt(s['cohorts'][c][v]['window_mean']['ADE15']) for c in ['old117','first82','old199','new97','new87_nonstatic','new_static10','all283_nonstatic']])
 oldrows=[]
 for v in VARIANTS:
  a=s['old_test_before_after'][v];d=a['per_seed_delta'];oldrows.append([v[0],fmt(a['before']['window_mean']['ADE15']),fmt(a['after']['window_mean']['ADE15']),f"{a['delta_ADE']:+.2f}",fmt(a['before']['window_mean']['FDE15']),fmt(a['after']['window_mean']['FDE15']),'/'.join(f'{d[str(k)]:+.2f}' for k in cfg['seeds']),sum(x<0 for x in a['per_recording_delta'].values())])
 seedrows=[[v[0]]+[fmt(s['reviewed'][v]['per_seed'][str(seed)]['ADE15']) for seed in cfg['seeds']] for v in VARIANTS]
 pairrows=[[k,f"{a['window_mean_delta']:+.2f}",f"{a['recording_equal_delta']:+.2f}",f"{a['recording_wins']}/13",'/'.join(f"{a['per_seed_delta'][str(seed)]:+.2f}" for seed in cfg['seeds'])] for k,a in s['paired'].items()]
 diag=[]
 for v in [VARIANTS[2],VARIANTS[5]]:
  diag.append([v[0],fmt(s['residual'][v]['reviewed']['ADE']),fmt(s['reviewed'][v]['window_mean']['ADE15']),fmt(s['reviewed'][v+'/frozen_diagnostic']['window_mean']['ADE15']),fmt(s['reviewed'][v+'/reference_diagnostic']['window_mean']['ADE15'])])
 cond=[]
 for k,c in s['conditions'].items():
  if c['hold_last'] is None:continue
  cond.append([k,c['hold_last']['windows']]+[fmt(c[v]['window_mean']['ADE15']) for v in VARIANTS+['hold_last']])
 recrows=[]
 for r in sorted({m['recording'] for m in meta}):
  recrows.append([r.split('_')[-1],sum(m['recording']==r for m in meta)]+[fmt(s['reviewed'][v]['per_recording'][r]['ADE15']) for v in VARIANTS+['hold_last']])
 reviewed_findings='\n\n'.join(f'{i+1}. **{item["title"]}** {item["text"]}' for i,item in enumerate(findings['points']))
 oldcomp=BASELINE/'summary.json'
 text=f'''# 第二轮扩充数据后的六组 Transformer 预测实验

完成日期：{datetime.now().date()}。新增 97 个严格筛选窗口已加入训练数据，主数据从 199 增至 296 窗，覆盖 13 个录制。已完成六组架构 × 13 组录制划分 × 3 个种子，共 234 个轨迹模型。复用 33 个排除集合完全匹配的相机模型，另训练 6 个相机模型。原数据、结果及选定基线均保留。

## 主要发现

{reviewed_findings}

## 六组模型和固定设置

| 组 | 历史信息交互 | 输出 |
|---|---|---|
| A | 行人与相机历史先拼接，再 self-attention | 图像轨迹 |
| B | 仅行人历史 self-attention | 图像轨迹 |
| C | 补偿行人历史独立 self-attention | 残差轨迹＋独立相机预测重投影 |
| D | 两路先 self-attention，再双向 cross-attention | 图像轨迹 |
| E | 两路先 self-attention，再行人 query 相机 | 图像轨迹 |
| F | 补偿行人与相机先 self-attention，再行人 query 相机 | 残差轨迹＋相机预测重投影 |

六组网络文件与上一轮逐字节一致，可训练参数约 21 万，组间差异 0.22%。输入 6 个历史点（0.5 秒），预测 15 个未来点（1.5 秒）。种子为 17、29、43；120 epoch × 2 updates、batch 48、AdamW 0.001、weight decay 0.01、40/20/40 的核查历史／核查前历史／扰动历史混合训练，以及录制均衡采样均保持原设置。全部按验证集图像 ADE 选检查点，没有根据测试结果调参。

C/F 与直接输出组同时改变了坐标、损失目标和几何合成，因此跨输出体系比较不是仅改变 attention 的消融。各组也没有加入新的 RGB 外观、深度或眼动特征。本轮不是 PPT 中带历史重建 GRU 模型的重训。

## 相同旧测试窗口的比较方式

原 11 组的测试／验证录制对应关系、具体验证窗口索引和相机 checkpoint 全部固定。新窗口如果属于该折验证录制，只保留为隔离数据，不参与该折训练或模型选择。两个新录制增加两组留出测试，验证录制按排序取下一项。同录制的全部人物和片段不跨训练／验证／测试。

这次原 199 窗的前后对比保留了测试标签、验证窗口、相机权重、网络和更新次数，改变的是允许进入训练的样本池。新增窗口在另一些折中被留出测试，不会把同一窗口的训练预测作为测试结果。相机使用完整有效录制，但同时排除该折测试和验证录制；没有使用全录制 refit。

## 全部 296 个窗口

{table(['方法','ADE ↓','FDE ↓','录制等权 ADE ↓','稀疏窗口 ADE ↓'],mainrows)}

单位为实验使用的 2560 × 1920 校正图像坐标中的像素，ADE 是未来 15 个点的平均中心误差。稀疏子集共 {s['sparse']['hold_last']['windows']} 窗，在每条连续区间内降低窗口重叠；跨人物、同录制和同事件依赖仍存在。

![不同测试范围](../../outputs/transformer_expanded_v2_20261005/presentation/cohort_results.png)

## 同一批旧 199 窗训练前后

{table(['组','加入前 ADE','加入后 ADE','ADE 差','加入前 FDE','加入后 FDE','三种子 ADE 差','改善录制数 / 11'],oldrows)}

差值为加入后减加入前，负数表示改善。这一表用于判断扩充训练的作用；不能把上一轮全部 199 窗的均值直接与本轮全部 296 窗的均值相减来代表提升。

![相同旧199测试窗](../../outputs/transformer_expanded_v2_20261005/presentation/same_old199_before_after.png)

## 数据子集分别统计

{table(['方法','原117','第一批82','旧199','第二批97','第二批去静止87','新增静止10','全部去静止283'],cohortrows)}

本表均为 ADE。全数据中有 13 个已标记静止人物对照（上批 3、本批 10），去静止列只排除这些已标记对照。原 117 的标签筛选比新增子集宽松，新旧子集难度也不相同。296 窗不代表 296 个独立行人；三个种子更不构成三倍独立数据。旧 199 行全部数组及元数据未改变。

## 随机种子和配对比较

{table(['组','种子17 ADE','种子29 ADE','种子43 ADE'],seedrows)}

{table(['比较 前减后','窗口平均差','录制等权差','改善录制数','三种子差'],pairrows)}

这些是预先定义的方案比较。最低测试均值只用于描述本轮结果，不自动替换已选定的部署基线；小差异需要结合种子和录制方向解释。

## C 和 F 的残差与相机诊断

{table(['组','残差 ADE','正常合成 ADE','保持朝向诊断','参考未来旋转诊断'],diag)}

诊断保持行人输出不变，只替换用于几何投影的旋转。参考未来旋转仅用于评估，不是正常推理成绩，也不保证严格数学上界。残差仍包含相机平移、视差、行人运动及标注噪声，不能称为纯粹物理行人运动。

## 条件和录制分析

{table(['条件','窗口数','A','B','C','D','E','F','保持位置'],cond)}

未来旋转分组只用于事后评分；这些未来量没有作为正常预测输入。G1 boundary 来自标注阶段的场景标记，other_classifications 包含原有样本和非边界候选，不等于人工确认的纯 G1。

{table(['录制末尾编号','窗口数','A','B','C','D','E','F','保持位置'],recrows)}

完整逐窗口、逐种子、输入扰动模式和预测时域曲线保存在 metrics.json、summary.json 与 presentation/horizon.png。

## 可视化和失败分析

{findings['visual_note']}

[本批固定10例六组同步视频](../../outputs/transformer_expanded_v2_20261005/presentation/fixed_second10_six_groups.mp4) · [原固定9例](../../outputs/transformer_expanded_v2_20261005/presentation/fixed_original9_six_groups.mp4) · [第一批固定9例](../../outputs/transformer_expanded_v2_20261005/presentation/fixed_first9_six_groups.mp4) · [本批事后改善与退化案例](../../outputs/transformer_expanded_v2_20261005/presentation/selected_second_gain_loss_six_groups.mp4)

固定案例在训练前选定；改善／退化案例按种子 17 的组间差事后选择，不能代表平均水平。白圈是 SAM 参考中心，橙圈是预测中心。预测在最后历史时刻一次生成，未来视频和 mask 只用于显示及评分，不会继续送入网络。

## 核验与保留

234 个 checkpoint、702 个预测文件全部重放；未来后缀扰动不改变正常输出，B/C 不读取相机特征，C/F 预测残差不受未来重投影旋转变动影响。独立几何实现最大投影差为 {geo['maximum_projection_difference_px']:.6f} 像素。上一轮 198 个模型在当前运行环境重放，最大差为 {cam['old_runtime_replay_max_px']:.6f} 像素；33 个复用相机权重哈希完全相同。

{ver['preserved_source_files']} 个受保护旧文件哈希未变；全部四个视频逐帧解码通过。数据来源仍是离线核查的自动标注和 Aria MPS 参考，不是独立人工 GT，也不是 RGB-only 或在线姿态可用性实验。保留所有未改善结果，没有因为测试误差删除窗口。
'''
 (REPORT/'RESULTS.md').write_text(text,encoding='utf8');print('Verified report written.',flush=True)
if __name__=='__main__':main()
