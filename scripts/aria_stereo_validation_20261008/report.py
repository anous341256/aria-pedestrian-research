"""Summarize the completed bounded test without treating consistency as accuracy."""
import json,csv
import numpy as np
from validate import ROOT,OUT,dump

frames=json.loads((OUT/'frame_metrics.json').read_text())
people=json.loads((OUT/'measurements.json').read_text())
tracks=json.loads((OUT/'track_summary.json').read_text())
env=json.loads((OUT/'environment.json').read_text())
stats=dict(frames=len(frames),clips=3,recordings=2,person_frames=len(people),
    measured_person_frames=sum(r['z_median_m'] is not None for r in people),
    inference_median_seconds=float(np.median([r['inference_seconds'] for r in frames])),
    inference_total_seconds=sum(r['inference_seconds'] for r in frames),
    stereo_max_skew_ms=max(r['stereo_skew_ms'] for r in frames),rgb_max_skew_ms=max(r['rgb_skew_ms'] for r in frames),clips_stats=[])
for cid in ['03','05','07']:
    rr=[r for r in frames if r['clip']==cid];dy=[];peaks=[];baselines=[]
    for f in range(21):
        meta=json.loads((OUT/cid/f'{f:03d}'/'metadata.json').read_text());dy+=meta['epipolar_abs_y_errors_px'];baselines.append(meta['baseline_m'])
        peaks.append(json.loads((OUT/cid/f'{f:03d}'/'inference.json').read_text())['cuda_peak_allocated_bytes'])
    stats['clips_stats'].append(dict(clip=cid,valid_fraction_mean=float(np.mean([r['valid_fraction'] for r in rr])),
        matched_feature_y_median_px=float(np.median(dy)),matched_feature_y_p90_px=float(np.percentile(dy,90)),
        matched_features=len(dy),baseline_median_m=float(np.median(baselines)),peak_cuda_allocated_GiB=max(peaks)/2**30))
dump(OUT/'summary.json',stats)
for name,rows in [('person_depth',people),('frame_metrics',frames),('track_summary',tracks)]:
    with (OUT/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Aria Gen 2 双目深度工具验证 — 2026-10-08','',
'已跑通 Meta 官方工具及 FoundationStereo 预训练模型。近处主要行人能取得连续、空间位置合理的深度估计；边缘视野和多人遮挡仍有缺失或异常。本次没有实测距离真值，不报告测距准确率。','',
'## 范围与方法','',
'固定原有片段 03、05、07 的前 21 个 RGB 时刻，共 63 对双目图像，来自两次录制。03/05 同属 192702；07 来自 164557。每段覆盖 0–2.0 秒、10 Hz。保持固定选择，没有替换失败或没有人物标注的时刻。',
'05 的已有 SAM 标注仅在原片段第 45–59 帧；本次选定前 21 帧没有已有 mask，因此该段仅验证全场景深度，不提供人物统计。03/07 合计 235 个已有 mask 人物帧，194 个至少包含 30 个通过一致性检查的内部深度点。14 个 clip-local ID 中 11 个至少有一帧可测；不是 14 个独立人物。','',
'流程：同步前方左右黑白图 → MPS 在线标定 + 双目极线校正 → FoundationStereo 视差 → Z = fB/d（米）→ 左右一致性检查 → 通过外参及两个时刻的 MPS 位姿映射到 RGB → SAM mask 内部取中位数。',
'采用官方大模型 vitl（23-51-11，epoch 40），512×512、32 次迭代，左右检查阈值 1 像素，没有微调或修改网络。基线约 14.23 cm。',
'RGB 使用已有 reference.npz 的 K 和 T_device_camera。CV→RGB 变换为 inv(T_world_device_RGB · T_device_RGB) · T_world_rectCV；不是把深度图直接缩放到 RGB。设备运动按时间补偿，但没有补偿这十几毫秒内行人自身的运动。',
'深度 Z 是沿相机光轴的距离；RGB 人物值是可见表面的深度中位数，不是身体中心三维位置。另存 range_median_m 表示到 RGB 光心的直线距离。mask 向内腐蚀 7 像素（15×15 核），测量不插值填洞；可视化点扩展仅用于显示。','',
'## 几何和运行检查','',
'|片段|左右特征纵向偏差中位数|通过检查的图像像素比例|','|---|---:|---:|']
for r in stats['clips_stats']:lines.append(f"|{r['clip']}|{r['matched_feature_y_median_px']:.3f} px|{r['valid_fraction_mean']*100:.1f}%|")
lines += ['',f"左右时间差最大 {stats['stereo_max_skew_ms']:.5f} ms；双目与 RGB 最近帧时间差最大 {stats['rgb_max_skew_ms']:.2f} ms。",
'特征匹配采用 SIFT 描述子比例及双向匹配，不用纵向误差预筛选。上述一致性比例包含天空、远背景，**不等于准确率或可用于训练的比例**。天空可能具有极大深度且通过左右检查。',
f"RTX 5070 Laptop 上，每对图像推理及左右复核中位耗时 {stats['inference_median_seconds']:.2f} 秒，63 对记录的推理合计 {stats['inference_total_seconds']:.1f} 秒；峰值分配显存约 {max(r['peak_cuda_allocated_GiB'] for r in stats['clips_stats']):.2f} GiB。不含模型加载、VRS 读取、RGB 映射、渲染；首帧包含冷启动，属于离线大模型处理速度。",'',
'## 人物连续性','',
'|片段/ID|可测/标注帧|深度范围(m)|每 0.1 秒绝对变化中位数(m)|最大相邻变化(m)|','|---|---:|---:|---:|---:|']
for r in tracks:
    if r['measured_frames']:
        lines.append(f"|{r['clip']}/{r['id']}|{r['measured_frames']}/{r['mask_frames']}|{r['depth_min_m']:.2f}–{r['depth_max_m']:.2f}|{r['median_abs_step_m']:.3f}|{r['max_abs_step_m']:.3f}|")
    else:lines.append(f"|{r['clip']}/{r['id']}|0/{r['mask_frames']}|无足够投影深度|—|—|")
lines += ['',
'近处主要目标 03/N01、07/N01、07/N02 的相邻变化中位数为 3.0、3.2、4.2 cm。它们支持“近处可见目标能得到连续深度”的初步判断；这些变化包含真实相对运动，不能称为测距误差或测量精度。',
'保留失败：03/N05–N07 在所选时刻位于有效双目投影视野外，没有足够深度点。07/N07 在第 3→4 帧由 9.36 m 变到 5.51 m，0.1 秒跳变 3.85 m；RGB 裁图显示该人较小且受前方人群遮挡，是需要复核的异常。尚未分离双目估计、跨相机遮挡/边界混合、mask 误差各自的责任。07/N06 也有 1.18 m 相邻跳变。没有为改善统计而删除这些结果。','',
'## 能支持什么','',
'可以把本工具作为离线近处行人深度候选来源，继续研究结合相机完整位姿恢复三维位置，以及处理相机平移。当前只能确认工具可运行、标定对齐合理及部分轨迹连续；不能据此认定它提供高精度人体中心 GT，也没有验证加入深度会提升轨迹预测。要测绝对精度，需要同场实测距离或独立深度参考。','',
'## 交付与审计','',
'- 当前展示：outputs/aria_stereo_validation_20261008/presentation/stereo_depth_review.mp4（半速，共 12.6 秒），以及 3 段单独视频、每段 0/1/2 秒截图和深度曲线。',
'- 原始浮点视差、深度：各帧 depth.npz；统一检查后的有效区域：checked_depth.npz；RGB 三维投影点：rgb_points.npz。大于 65.535 m 的值没有被 uint16 PNG 截断。',
'- 所有人物测量及汇总：person_depth.csv、track_summary.csv、frame_metrics.csv，以及对应 JSON。缺失保留为空。',
'- protocol.json 在推理前锁定时刻及参数；environment.json 保存版本和权重 SHA256；checks.json 保存源文件哈希与视频解码检查。',
'- 12 个已有 RGB/参考/标注文件哈希保持一致。原始 VRS/MPS 只读，未重新训练、改动 SAM 标注、预测模型或 PPT。',
'- 独立 /opt/aria-stereo-20261008 环境只读引用现有依赖，新安装包写入新环境；官方 vendor 代码保持原样。',
'- 实现修正：常量 255 图经双线性重采样会舍入成 254，支持区域由“恰好 255”修正为“至少 254”。旧 depth.npz 内的有效区域可能来自早期规则，正式统计统一读取 checked_depth.npz。原始视差和深度均保留。准备文件刷新期间一次推理读取受中断，准备完成后恢复；没有改变取样。',
'- initial_render_attempt/ 和 presentation_v1/ 为保留的中间显示版本；只看 presentation/。曲线不会跨缺失帧连线。','',
'## 官方来源','',
'- https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo',
'- https://github.com/NVlabs/FoundationStereo',
f"- 官方仓库 commit: {env['vendor_commit']}",f"- FoundationStereo commit: {env['foundation_commit']}",f"- 模型 SHA256: {env['weights_sha256']}"]
report=ROOT/'reports/aria_stereo_validation_20261008/RESULTS.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(stats,indent=2))
