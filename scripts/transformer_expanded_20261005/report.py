"""Reusable quantitative report; interpretation added after actual review."""
from shared import *

def main():
 s=read(OUT/'summary.json');verify=read(OUT/'verification.json');geo=read(OUT/'geometry_audit.json');camera=read(OUT/'camera_audit.json');videos=read(OUT/'presentation/video_checks.json');assert all(x['passed'] for x in [verify,geo,camera,videos])
 def fmt(x):return f'{x:.2f}'
 lines=['# 扩充数据后的六组 Transformer 实验 — 2026-10-05','',
 '完成 198 个轨迹模型：六种架构 × 11 个录制划分 × 三个种子。主数据为旧 117 窗＋新增严格筛选 82 窗，共 199 窗。相机部分复用 18 个排除集合完全匹配的旧模型，为新增五组排除集合训练 15 个模型。旧结果和选定 baseline 保留。','',
 '## 本轮结果如何理解','',
 '1. **扩充后，同一批旧测试窗口的六组平均 ADE 都下降。** A 从 148.83 降至 143.68，D 从 150.89 降至 143.75；B 的降幅最大，为 9.49 像素。A/B/D/E 在三个种子中都改善，C/F 的种子方向不完全一致。训练数据增加有帮助，但验证窗口也增加，不能归因为单一因素。','',
 '2. **A/B/D 已非常接近，没有稳定的架构胜者。** D 的全部窗口 ADE 最低，但仅比 B 低 0.22、比 A 低 0.58 像素；按录制等权则 B 最低。A 比 B 高 0.35 像素且种子方向不一致，因此此前相机特征融合的收益没有在扩充数据上稳定复现。D 比 A 在三个种子中均稍好，但只在 6/11 个录制中改善，不能据此声称普遍优势。','',
 '3. **显式几何合成仍受到未来相机预测误差限制。** C/F 总体落后于直接输出图像轨迹的组，F 相对 C 也没有总体收益。固定 C 的行人输出，将预测旋转换为参考未来旋转，ADE 从 163.74 降至 62.35 像素，说明当前未来相机误差是重要限制。局部片段仍有 C/F 更好的情况，不能将总体排序推广到所有场景。','',
 '4. **全部窗口上六组均未超过保持最后位置。** 保持最后位置 ADE 为 149.67，D 为 152.68；65 个稀疏窗口上 A/D 则略好于该基线，说明排序对采样和权重敏感。当前结果不能证明相机与行人信息无关联，也不能证明物理分解无价值；更直接的结论是：这批数据和训练设置下，相机特征交互尚未带来稳定增益，未来旋转预测仍需改善。','',
 '## 六组与本次不变的设置','',
 '| 组 | 时序交互 | 输出 |','|---|---|---|',
 '| A | 原始行人历史与相机历史特征先拼接，再 self-attention | 图像轨迹 |',
 '| B | 仅行人历史 self-attention | 图像轨迹 |',
 '| C | 补偿行人历史 self-attention，行人分支不读相机特征 | 残差＋独立相机预测重投影 |',
 '| D | 两路先 self-attention，再双向 cross-attention | 图像轨迹 |',
 '| E | 两路先 self-attention，再行人 query 相机 | 图像轨迹 |',
 '| F | 补偿行人历史与相机历史先 self，再行人 query 相机 | 残差＋独立相机预测重投影 |','',
 '网络文件与上轮逐字节一致，可训练参数仍约 21 万（差异 0.22%）。6 个历史点 / 0.5 秒，预测 15 个点 / 1.5 秒。种子 17、29、43；120 epoch × 2 updates、batch 48、AdamW 0.001、weight decay 0.01、配对扰动和录制均衡采样不变。全部按验证图像 ADE 选检查点。C/F 与原始输出组还改变坐标、损失目标和几何合成，跨输出体系不是单一 attention 因素的消融。','',
 '## 数据与划分','',
 '新增部分采用 82 个清晰连续窗口，而不是全部 180 个宽松候选。其中静止对照 3 窗，单独报告。旧 117 的标签和数组原样保留，其筛选比新增子集宽松，因此新旧子集难度和质量不相同。199 窗也不代表 199 个独立行人或 597 个独立种子样本。','',
 '原六个测试／验证录制对应关系保持不变，新增五个录制分别留出；同录制的全部片段和人物事件一起划分。相机训练用完整有效录制，但同时排除该折测试与验证录制；没有使用全录制 refit。原六折的相机 checkpoint 与旧实验完全相同。新增相机仍用原先 2,000 次更新、batch 256、验证旋转 MSE 选择。','',
 '原始 192702 录制占 101/199 窗，需同时看录制等权成绩。新增录制的测试窗口较少，部分验证集也只有 3 窗；当前仍是小样本实验。旧测试集前后比较中，测试窗口与相机权重相同，但训练数据及同一验证录制中的窗口有所增加，因此不是完全只改变训练样本数的单因素实验。','',
 '## 全部 199 窗','', '| 方法 | ADE ↓ | FDE ↓ | 录制等权 ADE ↓ | 稀疏窗口 ADE ↓ |','|---|---:|---:|---:|---:|']
 for v in VARIANTS+['hold_last','raw_cv']:
  a=s['reviewed'][v];lines.append(f'| {NAMES.get(v,v)} | {fmt(a["window_mean"]["ADE15"])} | {fmt(a["window_mean"]["FDE15"])} | {fmt(a["recording_equal"]["ADE15"])} | {fmt(s["sparse"][v]["window_mean"]["ADE15"])} |')
 lines+=['','![三种测试范围](../../outputs/transformer_expanded_20261005/presentation/cohort_results.png)','', '## 测试范围分别看','', '| 方法 | 旧117 ADE | 新82 ADE | 新79非静止 ADE | 静止3 ADE | 全部196非静止 ADE |','|---|---:|---:|---:|---:|---:|']
 for v in VARIANTS+['hold_last']:
  vals=[s['cohorts'][co][v]['window_mean']['ADE15'] for co in ['old117','new82','new79_nonstatic','static3','all196_nonstatic']];lines.append('| '+NAMES.get(v,v)+' | '+' | '.join(map(fmt,vals))+' |')
 lines+=['','不同测试集均值不能直接解释为扩充训练的收益。以下只比较同一批旧117测试窗。','', '## 相同旧117测试窗：训练前后','', '| 组 | 旧训练 ADE | 扩充后 ADE | 差值（新−旧） | 旧 FDE | 扩充后 FDE |','|---|---:|---:|---:|---:|---:|']
 for v in VARIANTS:
  r=s['old_test_before_after'][v];a=r['before']['window_mean'];b=r['after']['window_mean'];lines.append(f'| {v[0]} | {fmt(a["ADE15"])} | {fmt(b["ADE15"])} | {r["delta_ADE"]:+.2f} | {fmt(a["FDE15"])} | {fmt(b["FDE15"])} |')
 lines+=['','![相同旧测试窗口](../../outputs/transformer_expanded_20261005/presentation/same_old_test_before_after.png)','', '## 随机种子与关键配对','', '| 组 | 17 ADE | 29 ADE | 43 ADE |','|---|---:|---:|---:|']
 for v in VARIANTS:lines.append('| '+v[0]+' | '+' | '.join(fmt(s['reviewed'][v]['per_seed'][str(seed)]['ADE15']) for seed in [17,29,43])+' |')
 lines+=['','| 比较（前−后） | 窗口平均 ADE 差 | 录制等权差 | 录制改善数 / 11 |','|---|---:|---:|---:|']
 for pair,r in s['paired'].items():lines.append(f'| {pair} | {r["window_mean_delta"]:+.2f} | {r["recording_equal_delta"]:+.2f} | {r["recording_wins"]}/11 |')
 lines+=['','负值表示前一个方案误差更低；不同种子、录制的差值保存在 summary.json，不能将单次均值排序称为普遍结论。','', '## C/F 分支及相机诊断','', '| 组 | 残差 ADE | 预测相机后 ADE | 保持朝向诊断 ADE | 参考未来相机诊断 ADE |','|---|---:|---:|---:|---:|']
 for v in [VARIANTS[2],VARIANTS[5]]:lines.append(f'| {v[0]} | {fmt(s["residual"][v]["reviewed"]["ADE"])} | {fmt(s["reviewed"][v]["window_mean"]["ADE15"])} | {fmt(s["reviewed"][v+"/frozen_diagnostic"]["window_mean"]["ADE15"])} | {fmt(s["reviewed"][v+"/reference_diagnostic"]["window_mean"]["ADE15"])} |')
 lines+=['','替换旋转保持行人输出不变。参考未来旋转列是诊断，不能作为正常推理成绩，也不是严格数学上界；残差仍包含相机平移、视差、行人运动和标注噪声。','', '## 大转头与输入质量','', '| 条件 | 窗口 | A | B | C | D | E | F | hold |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for condition in ['turn_lt5','turn_5to15','turn_ge15','border','not_border']:
  co=s['conditions'][condition];lines.append('| '+condition+' | '+str(co[VARIANTS[0]]['windows'])+' | '+' | '.join(fmt(co[v]['window_mean']['ADE15']) for v in VARIANTS+['hold_last'])+' |')
 lines+=['','旋转分组来自参考未来量，仅用于事后评估，不送入预测网络。控制噪声和核查前历史的结果完整保存在 summary.json；离线核查标签和历史 MPS 不代表在线或 RGB-only 系统。','', '## 核验和可视化','',
 f'- 198 个检查点、594 个预测文件全部重放；未来后缀扰动不改变正常输出。B/C 不读取相机特征；C/F 的未来相机只影响重投影，不能改变已预测残差。',
 f'- 原108个模型在当前运行环境重放，旧预测最大差 {camera["old_runtime_replay_max_px"]:.6f} px；18个复用相机文件哈希完全相同。',
 f'- 独立 SciPy 几何核对最大差 {geo["maximum_projection_difference_px"]:.6f} px。',
 f'- {verify["preserved_source_files"]} 个旧模型、数据、源程序及原选定基线文件保持未变。',
 '- 固定旧9例、新9例在训练前确定；另有明确标记的事后改善／退化案例。所有视频逐帧解码通过。',
 '- 事后选择的四个新数据改善／退化窗口都来自 clip 14 的同一夜间场景，部分窗口重叠；不能作为四个独立事件，也不能代表整体表现。','',
 '[新9例六组同步视频](../../outputs/transformer_expanded_20261005/presentation/fixed_new9_six_groups.mp4) · [旧9例六组同步视频](../../outputs/transformer_expanded_20261005/presentation/fixed_old9_six_groups.mp4) · [新数据改善/退化案例](../../outputs/transformer_expanded_20261005/presentation/selected_new_gain_loss_six_groups.mp4)','',
 '预测在最后历史时刻一次生成，后续视频中的 RGB 和 SAM mask 仅用于展示与评分。预测点为 SAM mask 外接框中心；不声称真实三维行人运动。']
 REPORT.mkdir(exist_ok=True);(REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('Verified quantitative report saved.',flush=True)
if __name__=='__main__':main()
