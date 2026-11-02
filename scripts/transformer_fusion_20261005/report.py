"""Publish the completed experiment and inspectable case summaries."""
from shared import *


def main():
    s=read(OUT/'summary.json');cfg=read(OUT/'protocol.json');v=read(OUT/'verification.json');g=read(OUT/'geometry_audit.json')
    videos=read(OUT/'presentation/video_checks.json');selection=read(OUT/'presentation/selection.json')
    assert v['passed'] and g['passed'] and videos['passed']
    REPORT.mkdir(parents=True,exist_ok=True)
    def table(mode='reviewed',methods=None):
        lines=['| 方法 | ADE ↓ | FDE ↓ | 录制等权 ADE ↓ |','|---|---:|---:|---:|']
        for m in methods or VARIANTS:
            z=s[mode][m];a=z['window_mean']
            lines.append(f'| {NAMES.get(m,m)} | {a["ADE15"]:.2f} | {a["FDE15"]:.2f} | {z["recording_equal"]["ADE15"]:.2f} |')
        return '\n'.join(lines)
    cap=['| 组 | FF 宽度 | 可训练参数 | 活跃冻结参数 | 推理总参数 |','|---|---:|---:|---:|---:|']
    for m in VARIANTS:
        c=cfg['capacity'][m];cap.append(f'| {m[0]} | {c["ff_width"]} | {c["trainable_parameters"]:,} | {c["frozen_active_parameters"]:,} | {c["total_active_parameters"]:,} |')
    seedtable=['| 组 | seed 17 ADE | seed 29 ADE | seed 43 ADE | 39 个稀疏窗口 ADE |','|---|---:|---:|---:|---:|']
    for m in VARIANTS:
        z=s['reviewed'][m]['per_seed'];seedtable.append(f'| {m[0]} | {z["17"]["ADE15"]:.2f} | {z["29"]["ADE15"]:.2f} | {z["43"]["ADE15"]:.2f} | {s["sparse"][m]["window_mean"]["ADE15"]:.2f} |')
    bins=['| 未来 1.5 秒净旋转 | 窗口数 | A | B | C | D | E | F | 保持最后位置 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('turn_lt5','<5°'),('turn_5to15','5–15°'),('turn_ge15','≥15°')]:
        z=s['conditions'][name];bins.append('| '+label+' | '+str(z[VARIANTS[0]]['windows'])+' | '+' | '.join(f'{z[m]["window_mean"]["ADE15"]:.2f}' for m in VARIANTS+['hold_last'])+' |')
    cases=['| 事后选择案例（seed 17） | 对比 | ADE 变化 | 现象 |','|---|---|---:|---|']
    for key,a,b,desc in [('03_N01_015',VARIANTS[0],VARIANTS[3],'双向交互改善'),('09_N02_005',VARIANTS[0],VARIANTS[3],'双向交互退化'),
                         ('01_N01_025',VARIANTS[2],VARIANTS[5],'残差分支交互改善'),('07_N01_033',VARIANTS[2],VARIANTS[5],'残差分支交互退化')]:
        z=selection['scores'][key];cases.append(f'| {key} | {a[0]} → {b[0]} | {z[a]:.2f} → {z[b]:.2f} | {desc} |')
    report=f'''# 六组 Transformer 融合实验 — 2026-10-05

已完成 108 个轨迹模型：六种架构 × 六个录制划分 × 三个随机种子，复用 18 个冻结相机网络。
**按窗口平均，A 直接拼接最优（148.83 px）；增加 cross-attention 没有显示稳定额外收益。
六组 ADE 都未超过保持最后位置（141.43 px）。** 原来选定的旧版快照仍保留，没有因本轮测试排序自动替换。

## 六组到底做什么

| 组 | 输入与时序交互 | 输出 |
|---|---|---|
| A | 原始图像行人轨迹 + 相机历史特征先拼接，两层联合 self-attention | 未来图像轨迹 |
| B | 仅原始图像行人轨迹，两层 self-attention | 未来图像轨迹 |
| C | 旋转补偿后的行人轨迹，两层 self-attention；行人网络不读相机特征 | 行人残差 + 独立相机预测 → 几何重投影 |
| D | 两种历史各两层 self-attention，再同时进行双向 cross-attention，然后拼接 | 未来图像轨迹 |
| E | 两种历史各两层 self-attention，再由行人 query 相机 key/value | 未来图像轨迹 |
| F | 补偿后的行人历史与相机历史各两层 self-attention，再由行人 query 相机 | 行人残差 + 独立相机预测 → 几何重投影 |

F 的独立相机预测网络不读取行人特征。D 中更新的“相机 token”是融合表示，并不是单独输出的相机旋转预测。
所有 self/cross-attention 只读六个已观察的历史时间步。各组使用相同结构的输出层，但分别训练权重。
这六组不是严格 3×2 因子实验；B 没有相机分支。

![六组结构](../../outputs/transformer_fusion_20261005/presentation/architecture_six_groups.png)

## 数据与训练协议

- 沿用 117 个窗口、26 个连续片段、9 个有效短片、6 段录制；按录制分开训练、验证、测试。
- 观察约 0.5 秒的 6 个点，一次预测未来 1.5 秒的 15 个点，10 Hz。点为 SAM2 mask 外接框中心。
- 每组使用 120 epoch × 2 次更新、batch 48、AdamW lr 0.001、weight decay 0.01、梯度裁剪 1。
- 种子 17/29/43；按录制均衡采样，配对的训练样本及扰动：40% 已核查历史、20% 核查前历史、40% 加噪/丢点，最后历史点保留。
- 人体每步特征为位置偏移 2、邻接位移 2、有效位 1、时间 1；当前中心与 4 个归一化内参作为附加条件进入人体/联合 token。
- 相机每步特征为历史旋转向量 3 + 冻结相机 GRU 历史隐状态 32。B/C 不接收这些 token；C 的历史补偿本身仍使用 Aria 参考旋转。
- 相机源是匹配划分的 full-data GRU；没有重训相机、没有用 all-recording refit。A/D/E 使用编码器，C/F 的重投影使用完整相机预测器。
- 隐维 64、4 attention heads、dropout 0；共同解码结构 6×64 展平 → 128 → 30，输出相对最后观测点的偏移。没有历史重建损失、没有 flow 采样。
- 六组都使用归一化坐标未来 MSE；A/B/D/E 的监督是原始图像偏移，C/F 是旋转补偿偏移。全部按验证集最终图像 ADE 选择检查点，每 10 epoch 验证。
- 无效投影按 32,000 px 计分，没有删掉困难窗口；本轮六组正常推理无无效投影。

## 容量与可比性

{chr(10).join(cap)}

通过训练前仅依据参数公式调整 FF 宽度，匹配可训练容量，最大差约 0.22%；所有可训练参数都接入损失。
层数分配、FF 宽度和运算量仍不同，匹配参数数目不等于匹配 FLOPs。冻结预训练参数另计。
相同更新次数与数据预算不等于相同计算耗时。

A↔D 比较先融合与分开 self-attention 后双向交互；D↔E 比较双向/单向；C↔F 比较几何输出体系内的特征交互。
A↔B 比较相机历史与预训练编码信息的增量价值，不是单独比较某一层 attention。
跨原始输出/残差输出的对比同时改变坐标表示、监督目标和几何合成，不能把全部差异归因于 attention。
旧版还使用不同网络容量、flow 解码及训练目标；旧版比较是系统参照，不是严格结构消融。

## 总体结果

数值为像素；ADE 是 15 个未来时刻平均中心误差，FDE 是 1.5 秒终点误差。每个窗口先覆盖三个种子；不是把三个种子当成三倍独立数据。

{table(methods=VARIANTS+['hold_last','old_comp_dual','previous_separated','raw_cv'])}

![误差与预测时域](../../outputs/transformer_fusion_20261005/presentation/results_overview.png)

1. **相机特征有小幅增益。** A 相比 B：153.96 → 148.83，下降约 3.3%。三个种子方向一致，六段录制中四段改善，117 窗口中 74 个改善（先平均三个种子）。这支持在当前设置中利用相机历史信息，不代表所有融合方式都获益。
2. **先 self 再 cross 暂未超过直接拼接。** D 的 ADE 150.89、E 为 152.51。D−A 在三个种子分别为 +6.69、−0.57、+0.07；录制等权又略偏向 D，优势不稳定。不能据此说 cross-attention 普遍无效。
3. **F 对 C 的增益很小。** ADE 157.16 → 156.48（约 0.44%），三个种子有一个退化；录制等权只改善 0.23 px。FDE 224.58 → 218.24，但残差 ADE 70.26 → 71.38，未显示行人残差预测本身变好。
4. **排名取决于指标与样本权重。** A 是按窗口 ADE 最低；F 是六组中 FDE 和录制等权 ADE 最低。F 的 FDE 218.24 与保持最后位置 218.35 近乎相同，不表述为可靠超越。
5. **A 相对旧版 ADE 下降 5.8%，仍没超过简单基线。** A 在 117 窗口中只有 47 个平均 ADE 优于保持最后位置；这次架构比较没有证明复杂预测器优于该基线。

## 种子和减少窗口重叠

{chr(10).join(seedtable)}

39 个稀疏窗口中 A 仍是六组最低，但保持最后位置为 154.25 px，仍低于 A 的 161.06。
完整按轨迹、录制等权、逐种子、逐窗口结果保存在 summary.json / metrics.json；六段录制较少，不主张统计显著性。

## 什么场景失败

{chr(10).join(bins)}

分组采用参考未来净旋转，仅用于事后评估，不输入模型。≥15° 的 19 个窗口，A 相对 B 改善约 17 px，但六组仍均未超过保持最后位置。
有边界标记的 31 个窗口，A/B/C/D/E/F ADE 约为 196.32/195.70/204.17/196.07/195.59/205.76，保持最后位置 177.97。
大表观尺度（高度 ≥30%，43 窗口）中 C/F ADE 约 168.08/168.55，A 为 156.25；近处不自动意味着容易预测。
这里是误差分布，不是对“转头/边界导致失败”的独立因果证明。没有新增可靠深度、平移/视差、遮挡等级或行为标签，因此不虚构这些分组。

## 相机与残差诊断

| 组 | 残差坐标 ADE | 正常预测相机后的图像 ADE | 保持相机朝向的图像 ADE（诊断） | 使用未来参考旋转的图像 ADE（诊断） |
|---|---:|---:|---:|---:|
| C | 70.26 | 157.16 | 147.88 | 69.62 |
| F | 71.38 | 156.48 | 147.14 | 70.77 |

保持行人输出不变，替换旋转后误差大幅变化，说明几何输出组仍明显受未来相机预测限制。
参考未来旋转的列不能作为可部署预测成绩，也不是严格数学上界。只去除了旋转，残差还包含平移、视差、行人运动和标注噪声。
F 的图像均值小幅改善但残差略退化，不能将收益解释成“更准确解耦”；也不能据此断言网络有意学习误差抵消。

## 输入噪声检查

受控历史噪声下 ADE：A 149.70，B 154.74，C 159.42，D 152.19，E 153.81，F 158.79；保持最后位置 142.22。
核查前历史输入结果几乎不变，说明该输入版本在这批合格窗口上提供的扰动很有限，不能宣称真实跟踪噪声鲁棒性已验证。
本轮没有历史重建分支，因此不主张恢复/去噪能力。

## 同帧可视化与失败案例

固定视频：九个原来就确定的窗口，seed 17，六组放在同一帧上，半速播放共 37.8 秒。
另有 16.8 秒事后选择的改善/退化合集。白圈是 SAM 参考中心，橙圈是预测中心，连线表示误差。
视频中的未来画面和 mask 只用于展示与评分；预测起点之后模型不再读取观测。

{chr(10).join(cases)}

视觉核查：09_N02_005 终点画面中 D/E 比 A 向画面左侧偏离更多；07_N01_033 终点 C/F 虽接近参考中心，但整个路径 ADE 仍比 A/B 更大。这也提醒不能只看最后一张图判断 ADE。
固定案例 10_N01_020 中 A ADE 298.34，C 为 195.10，说明 A 的总体均值优势不适用于每个案例。
上述失败机制只陈述输出表现和可验证的替换诊断，不把 attention 权重当作贡献归因证据。

## 完整性检查

- 108 个模型、324 份三种输入下的预测文件全部复现；真实未来 xy、boxes、H、R、cam、seconds 改写后，正常输出变化为零。
- B/C 不接受相机特征 token；C/F 修改未来预测旋转不会改变行人残差输出。
- 相同相机历史的 57 对行人窗口，在每个 fold/seed 下相机输出差异为零。
- 独立 SciPy 旋转矩阵与逐射线重投影核对 C/F；最大投影差 {g['maximum_projection_difference_px']:.6f} px。原始 world-camera 姿态与历史 H 的坐标约定及往返也通过。
- {v['preserved_source_files']} 份原数据、标签、旧权重/输出和选定基线快照文件 SHA256 保持一致；没有覆盖旧方案。
- 两个视频逐帧解码通过（378、168 帧），图表和对比帧经过视觉检查。

## 文件入口与复现

- 方案：`plans/TRANSFORMER_FUSION_20261005.md`。
- 数据与折分：原 `outputs/biflow_aria_v1_20261002/dataset.npz`；本轮保存 `protocol.json`、`folds.json`、`source_hashes.json`。
- 权重/日志/预测：`outputs/transformer_fusion_20261005/fold_*/seed_*/[A–F]/`。
- 统计：本轮 `summary.json`、`metrics.json`、`residual_metrics.json`、`displacement.json`。
- 检查：`preflight.json`、`verification.json`、`geometry_audit.json`、`presentation/video_checks.json`。
- 展示：本轮 `presentation/START_HERE.md`、两个 mp4、13 张同帧对比图、结果图及结构图（PNG/PDF/SVG）。
- 源码：`scripts/transformer_fusion_20261005/`；`network.py` 定义六组，`run.py` 训练，`evaluate.py` 评分，`verify.py` 检查。

WSL `/opt/cvat-sam2/bin/python` 执行 run.py / verify.py；Windows `.conda-envs/phase1-py310/Scripts/python.exe` 执行 evaluate.py / geometry_audit.py / render.py / figures.py / report.py。
prepare.py 只在全新版本目录固定协议一次；训练可从本轮完成标记恢复，不允许悄悄修改协议后复用完成模型。
显示排序只是对已锁定六组的探索性比较，不在测试结果上调参；后续方案另行讨论。
'''
    (REPORT/'RESULTS.md').write_text(report,encoding='utf8')
    start='''# 六组 Transformer 对比

先看 [固定九例六组同步视频](fixed_9_six_groups.mp4)（37.8 秒）。
再看 [交互改善与退化案例](selected_gain_loss_six_groups.mp4)（16.8 秒，事后选择）。

布局：上排 A 先融合、B 仅行人、C 独立分离；下排 D 双向交互、E 单向交互、F 交互后重投影。
白圈是参考中心，橙圈是预测中心；面板 ADE 是该案例完整 1.5 秒路径误差，不是当前帧误差。
输入六个历史点，之后一次性预测十五个未来点，不接收后续观测。

- [完整实验报告](../../../reports/transformer_fusion_20261005/RESULTS.md)
- [六组结构图](architecture_six_groups.png)
- [结果图](results_overview.png)
- selection.json 包含全部案例分数、固定案例规则与事后选择规则。

按窗口 ADE：A 148.8，B 154.0，C 157.2，D 150.9，E 152.5，F 156.5。
保持最后位置 141.4；旧版 158.1。A 为本轮窗口均值最低，但没有自动替换既有选定基线。
'''
    (OUT/'presentation/START_HERE.md').write_text(start,encoding='utf8')
    write(OUT/'completed.json',dict(passed=True,models=108,windows=117,folds=6,seeds=[17,29,43],
                                   report=str((REPORT/'RESULTS.md').relative_to(ROOT)),no_test_tuning=True,old_baseline_preserved=True))
    print('Report and presentation entry saved')


if __name__=='__main__':main()
