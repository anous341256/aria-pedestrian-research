"""Render a concise evidence-backed local research report from saved results."""
from common import *
import collections

def main():
    s=read(OUT/'summary.json');h=read(OUT/'history_reconstruction.json')['summary'];diag=read(OUT/'diagnostics.json');cam=read(OUT/'camera_metrics.json');sel=read(OUT/'selection.json');wins=read(OUT/'paired_wins.json');cfg=read(OUT/'protocol.json');names=['hold','constant_velocity','raw_future','raw_dual','comp_future','comp_dual'];labels={'hold':'保持最后位置','constant_velocity':'原始匀速','raw_future':'原始轨迹，只预测未来','raw_dual':'原始轨迹，历史重建＋未来预测','comp_future':'旋转补偿，只预测未来','comp_dual':'旋转补偿，历史重建＋未来预测','comp_dual_reference_camera':'补偿双分支＋参考未来旋转','comp_cv_reference_camera':'补偿匀速＋参考未来旋转'}
    table='| 方法 | ADE 0.5 秒 | ADE 1.0 秒 | ADE 1.5 秒 | FDE 1.5 秒 | 录制等权 ADE 1.5 秒 |\n|---|---:|---:|---:|---:|---:|\n'
    for n in names:
        a=s['reviewed'][n];m=a['window_mean'];table+=f"| {labels[n]} | {m['ADE05']:.1f} | {m['ADE10']:.1f} | {m['ADE15']:.1f} | {m['FDE15']:.1f} | {a['equal_recording']['ADE15']:.1f} |\n"
    seedtable='| 方法 | 种子 17 | 种子 29 | 种子 43 |\n|---|---:|---:|---:|\n'
    for n in names[2:]:seedtable+='| '+labels[n]+' | '+' | '.join(f"{s['reviewed'][n]['per_seed'][str(i)]['ADE15']:.1f}" for i in cfg['seeds'])+' |\n'
    perrec='| 留出录制 | 窗口 | 保持位置 | 原始只预测 | 原始双分支 | 补偿只预测 | 补偿双分支 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for rec,num in sel['recordings'].items():perrec+=f'| {rec.rsplit("_",1)[1]} | {num} | '+' | '.join(f"{s['reviewed'][n]['per_recording'][rec]['ADE15']:.1f}" for n in ['hold','raw_future','raw_dual','comp_future','comp_dual'])+' |\n'
    history='| 输入条件 | 原输入可见点误差 | 原始双分支重建误差 | 补偿双分支重建误差 |\n|---|---:|---:|---:|\n'
    for mode,label in [('reviewed','当前回查版'),('pre_review','回查前配对轨迹'),('controlled_noise','人工抖动和缺失')]:
        a=h[mode];history+=f"| {label} | {a['raw_dual']['visible_input_MAE']:.2f} | {a['raw_dual']['reconstruction_visible_MAE']:.2f} | {a['comp_dual']['reconstruction_visible_MAE']:.2f} |\n"
    sections=f'''# 历史重建与未来预测联合模型第一版结果

2026 年 10 月 2 日。本轮在近处人物回查标注上实际训练了共享编码器、历史重建头与未来预测头，完成四组、六个录制划分、三个随机种子，共 **72 次轨迹模型训练**，另训练 18 个共享相机预测器。

**结果：旋转补偿版优于本轮原始轨迹网络，但历史重建没有形成明确的额外收益；所有学习版的总体平均误差仍高于保持最后位置基线。** 补偿版加入重建后 ADE 从 158.9 变为 158.1 像素，差异很小。原始坐标加入重建后从 192.7 增到 212.2 像素。以下均为留出录制上的结果，未根据测试分数继续调参。

## 观看入口

- [九段固定规则案例对比，37.8 秒](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/fixed_9_examples.mp4)
- [改善和失败案例，事后选择](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/selected_success_and_failures.mp4)
- [未来相机预测失败诊断](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/camera_failure_diagnostic.mp4)
- [结果图](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/results_overview.png)；[历史重建失败图](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/history_reconstruction_failure.png)

视频采用预先指定的种子 17，表格平均三个种子，二者数值不应混用。每个案例先播放 6 个历史点，再播放 15 个未来点，半速展示；预测起点后不使用新观测修正。白圈为 SAM 参考中心，橙圈为预测中心。普通案例选每段按 ID 排序第一名合格人物、时间居中的窗口；另外明确标出按分数事后选择的成功和失败例。05 号只有 15 帧的已标轨迹，不满足 21 帧要求，故没有 1.5 秒预测案例。

## 数据与划分

从十段标注中得到 **117 个窗口、26 个连续片段、9 段有效视频、6 个录制**。每窗输入 6 点（约 0.5 秒），预测 15 点（约 1.5 秒），滑动步长 5 帧。不跨缺失及 V6 不确定帧拼接，保留贴边窗口。补偿所需姿态检查没有额外排除窗口。

每轮一个录制测试、按固定顺序下一个录制验证、其余四个训练。六轮覆盖全部测试窗口一次；训练窗口数因划分在 38–106 之间。同录制全部人物与相邻片段留在同一组，避免按窗口随机切分。117 个重叠窗口不是 117 个独立实验，也不能由六个录制声称已覆盖广泛场景。

参考标签来自当前 SAM2 回查版，不是独立人工像素真值。历史标签由完整视频离线标注、历史 MPS 由闭环处理得到。本轮是**给定离线参考历史条件下的预测实验**，不是在线跟踪到预测的端到端成绩，也不是 RGB-only 推理结果。

## 实现的结构

本轮借鉴 [BiFlow 论文的双任务结构](https://arxiv.org/html/2510.00405v2#S4)，独立实现小型图像坐标版本。官方 [backbone](https://github.com/zoeyliu1999/EgoTraj-Bench/blob/3c519c8c15139e27109d1434f9d7e70f6f8823bc/models/backbone_biflow.py) 和 [flow matching](https://github.com/zoeyliu1999/EgoTraj-Bench/blob/3c519c8c15139e27109d1434f9d7e70f6f8823bc/models/flow_matching_biflow.py) 已核查真实代码，参考提交固定，原仓库保留在本地。

- 共享编码器：64 维 GRU，输入历史中心、框大小、有效标记和时间；结合绝对观察锚点、历史相机旋转与历史预测出的未来旋转。
- 历史重建头：条件 flow matching，输出过去 6 点。
- 未来预测头：相同类型解码器，输出未来 15 点。
- 两头共享编码器，训练损失为未来均方损失加 0.5 倍历史重建均方损失；对照将重建权重设为 0，网络尺寸不变。推理只运行未来头，历史头仅另行用于本报告的诊断。
- 这是逐人时序模型，没有复现论文的 BEV 输入、社会交互、专门的 EgoAnchor 意图模块或 K-to-K 候选注意力，也未使用论文预训练权重。不能将本轮结果当作原论文的复现分数。

四组均分配 121,450 个参数；正式预测路径均为 76,254 个参数。加入重建意味着额外的辅助训练梯度，不增加预测路径容量。每个划分／种子的未来相机模型先单独训练并冻结，四组获得同一份历史和预测相机条件；补偿组再显式做旋转坐标变换。相机模型不接收人物未来位置。

训练 120 轮，每轮两次 48 窗口抽样更新，按录制均衡采样；AdamW，学习率 0.001，权重衰减 0.01，梯度裁剪 1.0。每 10 轮以验证录制未来像素 ADE 选模。每次推理 10 个 flow 步，使用五组固定高斯起点，取五条输出均值作为单条报告轨迹，未按未来标签选择最优候选。测试时未来轨迹及未来姿态不进入预测分支。

## 预测结果

单位为 2560×1920 矫正图像中的像素，越小越好。学习版先在同一窗口上平均三个种子的误差，再汇总；录制等权列使大录制不因窗口较多占据主要权重。

{table}
补偿双分支相对补偿只预测，在 {wins['comp_dual_vs_comp_future']['improved']}/117 个窗口的三种子平均误差下降；小幅平均差异未证明稳定收益。保持最后位置仍优于四个学习版。原始坐标模型对初始化较敏感：

{seedtable}
各录制平均 ADE：

{perrec}
还保存了完整的按人物等权、贴边／非贴边和稀疏窗口统计。稀疏子集按同一连续片段每 25 帧选窗，避免该片段的历史加未来时段重叠，但不能消除同一画面多人的相关性。所有方法使用相同样本；本轮没有统计显著性宣称。

## 失败在哪里

### 历史重建没有学成可靠修复

单位为可见历史点相对回查标签的平均欧氏误差。当前回查历史作为自身参考时，输入误差按定义为 0，不能理解为真实物理位置完全无误差。

{history}
训练增强为 40% 当前回查历史、20% 身份匹配的回查前历史、40% 人工抖动及缺失。人工抖动标准差为图像宽／高的 0.5%，每个历史点有 20% 缺失概率，最后一点保留。输入同时提供有效性标记。

这些合格窗口中，回查前与回查后历史仅有 **1 个位置相差超过 1 像素**，实际修正配对几乎没有给出真实噪声监督。因此人工噪声测试单独报告。其可见输入点误差约 14.6 像素，重建后原始坐标约 62.4、补偿坐标约 40.6 像素，说明本版会改坏本来较可靠的历史。缺失点的平均补全误差也分别为 71.8 和 47.7 像素。

按划分和种子平均的诊断中，补偿双分支在训练组自身的带噪可见历史重建误差也约 {diag['averages']['comp_dual']['train']['history_visible_error']:.1f} 像素。故不能把失败完全归因于跨录制泛化；本版重建参数化、联合损失及按未来误差选模，都值得继续检查。这里只定位了未实现可靠修复，没有通过额外消融判定唯一原因。

### 未来相机预测仍然限制重投影

共享相机模型的平均旋转误差为 {cam['mean_degrees']:.2f}°，保持当前朝向为 {diag['frozen_camera_mean_angle_deg']:.2f}°，本轮学习相机预测没有超过这个简单参照。仅将补偿双分支重投影中的旋转换为未来参考值，ADE 从 158.1 降到 {s['reviewed']['comp_dual_reference_camera']['window_mean']['ADE15']:.1f} 像素。人物剩余轨迹预测保持不变，故这个对照直接显示未来相机误差的影响。

这个分支使用未来信息，只作为诊断。相同未来参考旋转条件下，匀速剩余轨迹预测 ADE 为 {s['reviewed']['comp_cv_reference_camera']['window_mean']['ADE15']:.1f} 像素，仍好于学习双分支。说明即使给出较准确相机旋转，当前学习版的人物预测也没有形成优势。旋转补偿后的剩余轨迹仍包含平移视差、尺度变化和标注噪声，不能视为纯粹物理行人运动。

### 第一版的结论和下一步

本轮完成了用户选择的双分支架构、分录制训练和结果交付，但没有证实历史重建的预测收益。下一版优先考虑保留可信观测、预测受限修正量与补全缺失点，并检查历史任务自身是否先达到有效修复；相机分支须保留冻结朝向强基线。扩充真实带噪／核查参考配对，比把当前高度重叠的 117 窗口继续扩成更多滑窗更有意义。这些是后续建议，本轮未据测试成绩实施新一轮调参。

## 核验与复现

- 修改未来点和未来姿态后，历史输入与正式预测输出变化为 0；未来推理不调用历史重建头。
- 历史重建损失确实向共享编码器传递梯度；四组网络容量相同。
- 旋转归一化／重投影往返最大误差约 0.000512 像素。
- 源标注、视频和参考姿态哈希核对通过，训练和评估组无录制交叉。
- 所有模型、逐轮训练记录、每个测试窗口及每个起始噪声样本均保存，没有只保存好例。

入口：[冻结协议](E:/New_research/outputs/biflow_aria_v1_20261002/protocol.json)、[数据划分](E:/New_research/outputs/biflow_aria_v1_20261002/folds.json)、[汇总统计](E:/New_research/outputs/biflow_aria_v1_20261002/summary.json)、[逐窗口指标](E:/New_research/outputs/biflow_aria_v1_20261002/metrics.json)、[关键核验](E:/New_research/outputs/biflow_aria_v1_20261002/verification.json)。

模型位于 `outputs/biflow_aria_v1_20261002/fold_*/seed_*/`，脚本位于 `scripts/biflow_aria_v1_20261002/`。数据准备之后依次运行训练、评估、诊断、视频和报告脚本。原数据与以往方案未覆盖。
'''
    report=ROOT/'reports/biflow_aria_v1_20261002/RESULTS.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text(sections,encoding='utf8')
    guide='''# 历史重建与未来预测第一版观看入口

这版已训练完成。旋转补偿优于本轮原始坐标网络；重建分支未显示明确额外收益，学习版整体仍未超过保持最后位置基线。

- [九段固定案例完整对比，37.8 秒](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/fixed_9_examples.mp4)
- [事后选取的改善与失败案例](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/selected_success_and_failures.mp4)
- [相机预测失败诊断，8.4 秒](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/camera_failure_diagnostic.mp4)
- [总体结果图](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/results_overview.png)
- [历史重建失败图](E:/New_research/outputs/biflow_aria_v1_20261002/presentation/history_reconstruction_failure.png)
- [完整结果与失败解释](E:/New_research/reports/biflow_aria_v1_20261002/RESULTS.md)

四格视频上排为原始坐标、下排为旋转补偿；左列只预测未来、右列加入历史重建。白圈是参考中心，橙圈是预测中心；画外预测在边界标叉。视频固定展示种子 17，半速播放，每个窗口先观察 0.5 秒，再冻结预测未来 1.5 秒。相机诊断右侧使用未来参考旋转，不能当作正式预测成绩。
'''
    (OUT/'presentation/START_HERE.md').write_text(guide,encoding='utf8');print(report)
if __name__=='__main__':main()
