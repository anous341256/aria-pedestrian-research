"""Build a traceable Chinese report from the completed, frozen evaluation."""
import json
from pathlib import Path
from prepare import ROOT, OUT, read, write

REPORT = ROOT / 'reports/factor_feasibility_expanded_20260928'
NAMES = {'151153': '站台、扶梯及站厅', '162538': '食品卖场', '164557': '街道及商场', '193709': '傍晚过街', '201129': '夜间站外'}
METHODS = ['stationary', 'raw_cv', 'comp_cv_predcam', 'comp_cv_refcam']
LABELS = ['保持最后图像位置', '原始轨迹＋匀速预测', '补偿轨迹＋历史外推未来相机（MPS 辅助）', '补偿轨迹＋参考未来相机（离线特权诊断）']

def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    s = read(OUT / 'summary.json')
    cases = read(OUT / 'candidates.json')
    rows = read(OUT / 'predictions.json')
    bg = read(OUT / 'background_summary.json')
    checks = read(OUT / 'package_verification.json')
    examples = read(OUT / 'examples.json')
    linkroot = '../../outputs/factor_feasibility_expanded_20260928/'
    raw = s['overall']['methods']['raw_cv']['ADE_1.5s']
    ref = s['overall']['methods']['comp_cv_refcam']['ADE_1.5s']
    newraw = s['new_only']['methods']['raw_cv']['ADE_1.5s']
    newref = s['new_only']['methods']['comp_cv_refcam']['ADE_1.5s']
    wins = sum(r['metrics']['comp_cv_refcam']['ADE_1.5s'] < r['metrics']['raw_cv']['ADE_1.5s'] for r in rows)
    samplewins = sum(r['metrics']['comp_cv_refcam']['ADE_1.5s'] < r['metrics']['raw_cv']['ADE_1.5s'] for r in examples)
    lines = [
        '# 同一时序预测器的相机旋转补偿可行性：扩大验证',
        '', '日期：2026-09-28。范围：离线机制验证；不训练新模型，不更改已有 SAM 标注。', '',
        f'**已完成 14 条轨迹案例，其中 10 条是新增案例，覆盖 5 段录制、381 个有效预测窗口。** 在使用 Aria/MPS 参考未来旋转的条件下，同一匀速预测器的 1.5 秒中心 ADE 从 {raw:.1f} 降到 {ref:.1f} 像素（下降 {(1-ref/raw)*100:.1f}%）；新增 10 例单独统计为 {newraw:.1f} → {newref:.1f} 像素（下降 {(1-newref/newraw)*100:.1f}%）。', '',
        '**这支持有条件的可行性：把相机旋转从轨迹中分开处理，具有可利用的预测价值。它尚未证明仅靠历史 RGB 的完整预测系统有效。** 用历史角速度外推未来相机时，整体误差反而增大。未来相机运动仍是当前方案没有解决的环节。', '',
        '这里的“14 例”是 14 个录制内 SAM ID 的连续片段，不代表 14 位不同的人或 14 个独立场景。同伴可能跨视频重复出现，重叠窗口也不独立。', '',
        f'[查看全部案例对比图]({linkroot}comparison_14_cases.png) · [图表 PDF]({linkroot}comparison_14_cases.pdf)', '',
        '## 实验保持了什么', '',
        '所有分支使用相同的 6 个历史中心点（原生 10 Hz，跨度约 0.5 秒）、相同的最小二乘匀速预测器、相同的 15 个未来目标帧（1.5 秒）。以最后观测点为锚点，按历史时间戳拟合速度，每 5 帧建立一个预测窗口。没有训练、调参或增加预测网络容量。', '',
        '中心点来自 SAM mask 的可见部分外接框，不是人体三维根节点。评价均回到原始 2560×1920 图像中，计算二维中心误差；ADE 是整个预测时段的平均误差，FDE 是最后一帧误差。补偿空间中的曲线只作解释展示，不与原始像素误差混算。', '',
        '| 分支 | 操作与信息条件 |', '|---|---|',
        '| 保持位置 | 所有未来时刻均保持最后一个图像中心。用于检查原始匀速基线是否过弱。 |',
        '| 原始匀速 | 直接外推原始图像中心序列。 |',
        '| 补偿＋外推相机 | 用历史 MPS 旋转归一化轨迹，再用同一匀速预测器；相机未来旋转由历史旋转向量的斜率外推。 |',
        '| 补偿＋参考未来相机 | 行人部分的预测与上一分支完全相同；只把重投影旋转替换为未来 MPS 参考值。它使用未来信息，是离线特权诊断，并非可部署模型，也不是严格数学上界。 |', '',
        '令 R_i 为相机到世界的旋转，K 为线性 RGB 内参。历史点通过 H(i→t)=K R_tᵀ R_i K⁻¹ 变换到预测起点的相机朝向；匀速外推后，通过 K R_futureᵀ R_t K⁻¹ 投回未来图像。这里只补偿旋转，没有恢复深度或平移，剩余轨迹仍包含视差、行人运动、遮挡和标注噪声。', '',
        '**MPS 是已有的离线闭环估计，不是姿态真值；历史 MPS 也可能受后续观测影响。** 因而两个 MPS 辅助分支都不能称为因果 RGB-only 方法。代码没有把未来点或未来旋转传入历史预测函数，但这不等于证明上游闭环 MPS 是在线因果的。未来 RGB 只用于可视化核对。', '',
        '本对照同时改变了历史表示与未来重投影信息，因此不能把全部收益归因于“历史补偿本身”，也不能把误差降幅解释成相机运动对总运动的物理贡献比例。', '',
        '## 样本如何确定', '',
        '遍历 5 段已标注视频的全部 25 个 SAM ID。每个 ID 选取最长连续非空片段，要求至少 41 帧；长度相同时选较早片段。得到 18 个候选，在计算新增预测分数前按 1 Hz 加首尾帧检查身份与明显 mask 错误，保留 14 个。原有 4 例沿用原范围和已有抽样检查；另外 14 个候选共检查了 26 张联系表。', '',
        '评分前排除了 4 个候选：151153_S3、151153_S5 存在身份转移或背景碎片；164557_S3 转移到另一已跟踪成人并重复计数；164557_S5 末尾转移到遮挡者及背景。排除的是事先选定的整个最长片段，没有按预测分数裁剪成较好片段。其余 7 个 ID 未达到长度条件。', '',
        '保留片段仍有贴边、模糊及框几何噪声。例如 151153_S2 的起始框包含邻近人物。检查由助手抽样完成，并非用户逐帧确认的金标准。', '',
        'MPS 时间对齐要求插值两端距 RGB 时间不超过 5 ms、质量至少 0.5、窗口内处于同一图。151153_S1 的前 4 个候选窗口缺少有效参考位姿，对所有分支一并排除。该案例仅余 2 个有效窗口，不能单独支撑强结论。所有排除均记录在数据文件中。', '',
        f'[候选与人工式抽样检查记录]({linkroot}candidate_review.json) · [全部 ID 与连续片段清单]({linkroot}candidate_inventory.json) · [最终案例范围]({linkroot}candidates.json)', '',
        '## 总体结果', '',
        '单位为原图像素，越小越好。主表按窗口平均；所有方法评价同一批窗口。', '',
        '| 方法 | ADE 0.5 秒 | ADE 1.0 秒 | ADE 1.5 秒 | FDE 1.5 秒 | 新增 10 例 ADE 1.5 秒 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for m, label in zip(METHODS, LABELS):
        a = s['overall']['methods'][m]
        lines.append(f"| {label} | {a['ADE_0.5s']:.1f} | {a['ADE_1.0s']:.1f} | {a['ADE_1.5s']:.1f} | {a['FDE_1.5s']:.1f} | {s['new_only']['methods'][m]['ADE_1.5s']:.1f} |")
    stationary = s['overall']['methods']['stationary']['ADE_1.5s']
    lines += ['', f'参考未来相机分支相对保持位置基线也改善了 {(1-ref/stationary)*100:.1f}%，并非只胜过原始匀速这一较弱基线。新增 10 例共 176 个窗口；旧 4 例共 205 个窗口。旧例逐点重算结果与原实验完全一致。', '',
        f'按完整轨迹平均，参考未来相机分支在 **14/14 条轨迹、5/5 段录制**上优于原始匀速；按窗口为 **{wins}/381 改善，{381-wins} 个未改善**。新增部分为 162/176 改善。固定展示窗口只有 {samplewins}/14 改善，不能把轨迹平均结果说成每个展示例都改善。', '',
        '## 14 个案例逐例结果', '',
        '帧号为视频中从 0 开始的帧号。片段范围是完整评测来源，不是只评展示窗口；每行链接打开对应的固定示例图。数值均为该轨迹全部有效窗口的 1.5 秒 ADE。', '',
        '| 案例 | 场景 | 来源 | 帧范围 | 窗口数 | 保持位置 | 原始匀速 | 补偿＋外推相机 | 补偿＋参考未来相机（离线） |',
        '|---|---|---|---|---:|---:|---:|---:|---:|',
    ]
    for c in cases:
        a=s['per_case'][c['key']]
        image=f"{c['case_number']:02d}_{c['key']}.png"
        vals=' | '.join(f"{a['methods'][m]['ADE_1.5s']:.1f}" for m in METHODS)
        lines.append(f"| [{c['case_number']:02d} · {c['key']}]({linkroot}examples/{image}) | {NAMES[c['recording']]} | {'原例' if c['previous_pilot'] else '新增'} | {c['start']}–{c['end']} | {a['windows']} | {vals} |")
    lines += ['', '固定示例按“尽量接近片段开始后第 50 帧的有效预测起点”选择，短片段取最近可用点；不是按误差挑选。图中显示当前帧、未来参考帧、原始及补偿后的横坐标序列。左右曲线共用尺度，表内 ADE 同时考虑横纵坐标。画外预测只在图片边缘标示，计算指标时未裁剪。', '',
        f'[案例 01–06 联系表]({linkroot}example_contact_1.jpg) · [案例 07–12 联系表]({linkroot}example_contact_2.jpg) · [案例 13–14 联系表]({linkroot}example_contact_3.jpg)', '',
        '**保留的失败示例：** 食品卖场 162538_S2，固定起点第 293 帧。原始匀速 ADE 664.7 px，参考未来相机 667.4 px，略微变差。归一化后的未来轨迹仍会弯折，说明补偿旋转后剩余运动也不一定适合匀速外推；仅凭此例不能确定弯折具体由行人动作、相机平移还是框噪声造成。', '',
        '## 结果是否依赖某种统计口径', '',
        '| 统计口径 | 样本量 | 保持位置 ADE | 原始匀速 ADE | 补偿＋外推相机 ADE | 补偿＋参考未来相机 ADE |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for key,label in [('equal_case','每条轨迹等权'),('equal_recording','每段录制等权'),('nonoverlap','同轨迹不重叠目标窗口'),('new_nonoverlap','仅新增、不重叠目标窗口'),('no_edge','所有历史及目标框均离边界 >5 px'),('all_methods_projectable','四分支均可投影的共同子集')]:
        d=s[key]
        count=14 if key=='equal_case' else 5 if key=='equal_recording' else d['windows']
        methods=d if key.startswith('equal_') else d['methods']
        vals=' | '.join(f"{methods[m]['ADE_1.5s']:.1f}" for m in METHODS)
        lines.append(f'| {label} | {count} | {vals} |')
    lines += ['', '不重叠采样使用固定 20 帧步长；只消除同一轨迹的目标时段重叠，不能消除同录制不同人物共享相机运动的依赖。等权与各子集均支持参考未来旋转条件下的改善，但本次不作统计显著性或总体泛化宣称。', '',
        '历史外推相机分支在 45/381 个窗口出现不可投影步骤，合计 216/5715 个未来点。主指标对这些点采用预先约定的 32000 px（10 倍图像对角线）惩罚；有限但超出画面的预测不裁剪，接近投影奇点时可能出现很大误差。**即使只看全部分支可投影的 336 个共同窗口，外推相机仍比原始匀速差：664.7 对 508.7 px。**', '',
        '另用不依赖像素无效点惩罚的观察射线夹角误差复核：', '',
        '| 方法 | 平均角误差（度） | 终点角误差（度） |', '|---|---:|---:|',
    ]
    for m,label in zip(METHODS,LABELS):
        a=s['angular_overall'][m]
        lines.append(f"| {label} | {a['mean_deg']:.2f} | {a['final_deg']:.2f} |")
    lines += ['', '像素与角度两种口径都表明：目前这个简单的未来角速度外推不足以实现收益。不能把失败完全解释为无效投影惩罚造成的。', '',
        '按未来 1.5 秒相对起点的最大旋转角事后分组（该分组不输入预测器）：', '',
        '| 旋转幅度 | 窗口数 | 原始匀速 ADE | 补偿＋参考未来相机 ADE |', '|---|---:|---:|---:|',
    ]
    for key,label in [('lt5','<5°'),('5to15','5°–15°'),('ge15','≥15°')]:
        a=s['rotation_strata'][key]
        lines.append(f"| {label} | {a['windows']} | {a['methods']['raw_cv']['ADE_1.5s']:.1f} | {a['methods']['comp_cv_refcam']['ADE_1.5s']:.1f} |")
    lines += ['', '**低旋转组仅有 2 个窗口，而且补偿分支更差。** 当前样本明显偏向转动较大的片段，无法据此证明在稳定相机、所有距离、所有遮挡程度及行人动作下都有效。片段长度门槛还会偏向 SAM 持续跟踪成功的目标。', '',
        '## 背景运动的独立核对', '',
        f"在各保留范围每隔 10 帧采样相邻帧对，经既定 KLT 双向跟踪及相似性筛选，共 {bg['sampled_pairs']} 对候选、{bg['valid_pairs']} 对有效、{bg['points']:,} 个匹配点。尽量排除已有 SAM 人物框后，先在每个帧对计算背景位移中位数，再在帧对间取中位数：", '',
        '| 处理 | 帧对中位位移的中位数（像素） |', '|---|---:|',
    ]
    for key,label in [('raw','原始'),('rotation','正确旋转补偿'),('inverse_rotation_control','逆向旋转负对照')]:
        lines.append(f"| {label} | {bg['median_pair_median_px'][key]:.2f} |")
    lines += ['', f"{bg['fraction_pairs_rotation_better']*100:.1f}% 的有效帧对在正确补偿后残差下降。这支持坐标变换方向和相机旋转解释具有实际作用。背景候选并非人工确认的静态真值，未标注行人或其他运动物体可能残留；该结果也不证明已完全分离相机平移和行人运动。", '',
        f'[背景完整结果]({linkroot}background_summary.json) · [逐帧对记录]({linkroot}background_pairs.json)', '',
        '## 核验、复现与数据记录', '',
        '- 恒等旋转下，两种坐标表示的预测一致；已检查旋转正负方向与往返变换。',
        '- 已知旋转与匀速参考平面轨迹的合成测试误差约 10⁻¹² px。',
        '- 改变输入序列的未来后缀，不改变历史预测函数的输出；只改变使用未来参考旋转的诊断重投影。',
        f"- 独立从预测坐标重算 ADE/FDE，最大差为 {checks['independent_metric_max_difference_px']:.1f} px；原有 4 例逐点预测最大差为 {checks['pilot_prediction_max_difference_px']:.1f} px。",
        '- 原预测函数源码、原实验汇总和原 SAM 标注哈希核对通过；源 VRS 的大小及修改时间未变。',
        '- 本轮的协议、候选审查、所有预测坐标、排除原因和 14 张示例图均已保存。', '',
        f'[冻结协议]({linkroot}protocol.json) · [合成与输入边界检查]({linkroot}verification.json) · [交付核验]({linkroot}package_verification.json) · [完整统计]({linkroot}summary.json) · [逐窗口预测]({linkroot}predictions.json)', '',
        '协议元数据说明：protocol.json 从原 4 例协议继承的 limitations 首项仍写着“4 tracks / 3 recordings”，这是未同步的旧样本量文字。本次实际范围以同文件 selection / expansion_rule、14 个 candidates 和 summary 为准。为保留评分前协议及其哈希，没有事后改写该文件；已单独保存 metadata_errata.json 澄清。该文字错误不影响任何计算。', '',
        f'[协议文字勘误]({linkroot}metadata_errata.json)', '',
        '脚本位于 scripts/factor_feasibility_expanded_20260928/。prepare.py 生成候选及审查图；review_decisions.json 保存评分前检查；run.py 调用未修改的旧预测核心；background_check.py 检查背景；deliver.py 输出图及独立核验；report.py 由结果生成本文。run.py 在 summary.json 已存在时主动拒绝覆盖，复现实验须先改到新的输出目录，不能删除已有结果来重跑。', '',
        '```powershell',
        'wsl.exe -e /opt/aria-vrs/bin/python /mnt/e/New_research/scripts/factor_feasibility_expanded_20260928/report.py',
        '```', '',
        '## 与已有研究及论文表述的关系', '',
        'Neumann 与 Vedaldi 的 CVPR 2021 工作提供了相关先例：把行人和自车运动分开处理，并将预测变换回未来相机视角。其场景是车载，预测器使用视觉特征等信息，并非本实验的坐标匀速预测器；本次只借鉴其几何分解思路，不声称复现该方法。', '',
        '[Neumann & Vedaldi, Pedestrian and Ego-Vehicle Trajectory Prediction From Monocular Camera, CVPR 2021（论文原文）](https://openaccess.thecvf.com/content/CVPR2021/papers/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf)', '',
        '可据本次结果作如下表述：', '',
        '> 在五段第一人称录制的十四条经抽样检查的 SAM 轨迹片段上，我们固定短历史匀速预测器，比较原始图像轨迹与旋转归一化轨迹。在利用 Aria/MPS 参考未来旋转重投影的离线诊断条件下，1.5 秒中心 ADE 从 613.9 降至 179.5 像素；新增十条片段上得到相同方向的改善。结果支持显式处理相机旋转的潜在价值，但基于历史旋转的简单外推尚未带来端到端改善，未来相机预测及纯 RGB 估计仍需验证。', '',
        '本轮已达到“扩大到至少十个案例，做简单可行性分析”的范围。没有由此推出纯 RGB 部署效果、真实物理运动的唯一分解、或相机与行人运动统计独立。', '',
    ]
    (REPORT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write(OUT / 'metadata_errata.json', {
        'type': 'post-evaluation text correction; frozen protocol preserved',
        'field': 'protocol.json limitations[0]',
        'inherited_text': '4 tracks / 3 recordings',
        'actual_expanded_scope': '14 recording-local identity segments / 5 recordings, including 10 new cases; 381 windows',
        'note': 'Selection and expansion fields in frozen protocol already specify expanded scope. No evaluation inputs, outputs, thresholds or frozen hashes changed.'
    })
    print('REPORT', REPORT / 'RESULTS.md')
    print('COUNTS', {'cases': len(cases), 'windows': len(rows), 'improved_windows': wins, 'improved_fixed_examples': samplewins})

if __name__ == '__main__':
    main()
