"""Persist annotation handoff, counts and project memory from verified artifacts."""
from qa import *

def main():
 s=read_json(OUT/'dataset/summary.json');t=s['totals'];audit=read_json(OUT/'dataset/package_audit.json');assert audit['all_checks_passed']
 report=ROOT/'reports/sam2_g1_expansion_20261005';report.mkdir(exist_ok=True)
 events=[e for cid in CLIPS for e in read_json(OUT/cid/'final/review.json')['events']];ec=collections.Counter(e['action'] for e in events);accepted=sum(e['action']=='prompt' and e['accepted'] for e in events)
 rows=['| 片段 | 场景 | 秒 | ID | mask | 连续轨迹¹ | 清晰窗口¹ | 旧口径窗口 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in s['clips']:rows.append(f'| {r["clip"]} | {r["scene"]} | {r["seconds"]:g} | {r["people"]} | {r["masks"]} | {r["continuous_runs_21"].get("clear",0)} | {r["windows"].get("clear",0)} | {r["windows"].get("compatible",0)} |')
 text='''# G1 近处连续标注扩充 — 2026-10-05

完成 10 个新增时间区间，合计 70 秒、700 帧，来自 9 个录制。其中 5 个录制首次进入当前近处人物标注集。与旧 53 秒数据没有源时间重叠；合计 20 段、123 秒、11 个录制来源。这里的“新增”是扩充现有标注集，不是新采集视频；相机全数据训练此前已经读取过部分新增录制。

最终保存 **67 个片段内 ID、88 个标注区间、2,186 个 mask**。不是 67 个独立新人物，同录制相邻事件可能涉及相同人物。保留 **50 个待核查人物帧**，另有 **60 个带 UNCERTAIN 状态的 mask**；不补插值 mask、不跨遮挡硬连身份。

## 可用于后续预测的数据

- **严格清晰子集：21 条至少 21 点的连续轨迹，82 个窗口**，覆盖 9 个片段。每窗 6 个历史点 + 15 个未来点，10 Hz，步长 5 帧；历史跨度 0.5 秒，预测跨度 1.5 秒。其中 3 窗是静止人物对照，79 窗是其余行走候选；静止对照应单列。进一步稀疏取窗得到 26 个互不重叠窗口（同人物分段内），不同人物可能共享画面，仍非完全独立样本。
- **旧筛选口径：新增 180 窗，旧 117 窗合计 297 窗**。旧口径只排除缺失及 UNCERTAIN，仍包含边缘、局部身体、骑行交互等不适合直接视作清晰 G1 的区间。这个数用于数量可比，不能写成 297 个高质量新样本。
- 严格子集还排除：已核查的身体遮挡、只露头部、骑行及台阶边界、图像边界截断、过小目标、mask 与人物提示框明显不一致。每一帧的理由均已保存。窗口步长从各自连续段起点重新开始，因此 82 个清晰窗口中部分起点不同于旧口径；不是简单筛选旧数组中的 82 行。
- **15 号没有通过严格连续窗口筛选**。原始录制中人物太近导致身体出画、SAM 部件分割和台阶背景遮挡；保留全部预览及修正记录，但不为增加数量将其纳入推荐子集。

'''+ '\n'.join(rows)+'''

¹ 连续轨迹与清晰窗口按本次严格规则统计；短时进出人物仍保存在标注中，不因不能组成预测窗口而删除。G1 是目标条件下的候选筛选：14/15/20 保留边界标记，19 为静止对照；不能把每一条 mask 都宣称为严格 G1。

## 标注流程及质量处理

沿用上一版：12% 图像高度进入、8% 保持（图像尺度，不是米制距离）、无人数总上限、每批最多三人；逐帧检测与分块补查→逐人核查检测编号→清晰锚点→冻结 V6/SAM2 大模型双向传播→前批同帧可靠 mask 负点→核查及单独记录提示修正。

初筛 11 段共 75 秒。17 号在矫正后画面核查中呈现扶梯口拥挤，不纳入本次 G1 扩充，保留输入及发现结果。火车玻璃倒影、广告、手机和骑行目标另行排除。检测器编号只作为候选，不当作人物身份：已拆分多个从前景人跳到后方人的编号。

发现并记录的主要失败：

1. 12 号交叉经过、柱后出现时误贴前景人的头部；14 号小片段误贴前景人的头/腿。撤销不能确认的 mask，保留断点。
2. 13 号白背心人物的局部帧只分出皮肤/手臂；补提示及短区间重锚后仍有不稳区间，43–50 帧排除预测。另一人的 mask 带上旁边人物碎片，重新提示后检查边界和中心。
3. 15 号近处成人从整人变成皮肤或衣服部件。补提示改善了多数可见区域，但受裁切和部件分割影响，严格子集不收该片段的窗口。
4. 18 号栏杆后人物与前景交叉者只局部可见；一次修正反而贴到前景行人，最终撤销该帧。栏杆、局部身体和静止对照均有独立标记。

未对 mask 做相减、裁剪拼接或用插值制造连续性。最终 mask 仍是经过代理视觉核查的离线自动参考，尚非独立人工逐像素 GT。双向 RGB、Aria 参考旋转和区间核查用于离线标注；本次不构成在线跟踪、RGB-only 推理或负点单独提升准确率的实验证据。修正帧使用独立图像提示，原始传播记忆保留用于审计，不能声称整条链已用修正后记忆重跑。

## 核验与文件

- 所有 mask 的 RLE、像素数、外接框、人物/帧唯一性检查通过。
- 最终 1,300 个实际使用的负点均落在同帧、不同 ID 的最终 donor mask 内；撤销 donor 的依赖作保守隔离。
- 按 640×480 检查交集大于等于 10 像素且占较小 mask 15% 以上的重叠告警最终为 0。这是机械检查，不是身份准确率。
- 相机参考对齐、图连接、时间步及投影有效性通过；新增区间没有历史姿态缺失填补。
- 252 个原有标签/协议/核心程序文件哈希未改变；联合数组的前 117 行与旧实验数组逐项完全相同。
- 10 个完整视频及 70 秒合集全部逐帧解码通过。原始自动输出、各轮修正和失败均保留。

文件入口：

- `outputs/sam2_g1_expansion_20261005/presentation/START_HERE.md`：完整视频。
- 每段 `final/`：交付 mask、决策、census、核查与修正日志。片段根目录是原始自动输出；`review_v1`、`review_before_delivery` 等为保留的中间版本。
- `dataset/clear_dataset.npz` + `clear_samples.json`：推荐 82 窗，数组含 xy、boxes、R、K、H、cam、seconds；未来值是监督目标或诊断参考，不是推理输入。
- `dataset/combined_compatible_dataset.npz` + `combined_compatible_samples.json`：旧口径 297 窗，前 117 行原样保留。
- `dataset/continuous_runs.json`、`frame_quality.json`：连续区间、每帧质量标记及剔除原因。
- `dataset/summary.json`、`integrity.json`、`package_audit.json`：统计和核验。

未启动任何新预测模型训练。下一轮实验必须按录制分组，并保留同人物事件关系；旧相机模型已训练过部分新增录制，不能直接冒充这些新样本的 held-out 相机模型。旧模型、旧分割与原始 117 窗成绩保持不变。
'''
 text+=f'\n本轮第一阶段核查修正日志：{ec["prompt"]} 次显式提示尝试、{accepted} 次通过该轮规则，另有 {ec["revoke"]} 次撤销；后续重锚与最终撤销另记，最终数量以上表及数据包为准。\n'
 (report/'RESULTS.md').write_text(text,encoding='utf8')
 entries=['# 新增 G1 近处标注：观看入口','', '**先看 [70 秒完整合集](E:/New_research/outputs/sam2_g1_expansion_20261005/presentation/ALL_10_CLIPS_70s.mp4)**。共 10 段，原速 10 fps，颜色和编号保持片段内人物身份。','', '画面上的 partial / review 表示局部身体、遮挡、边界或其他质量条件，不直接进入清晰连续窗口。未显示 mask 的已登记区间保留为待核查。','', '可以先看 11（站台）、16（商场通道）、21（迎面行人）；13 包含明确记录的部件分割失败。15 本次没有通过严格连续窗口筛选，仍保留完整视频便于核查。','', '| 片段 | 单独视频 | 清晰窗口 |','|---|---|---:|']
 for r in s['clips']:entries.append(f'| {r["clip"]} | [{r["scene"]}](E:/New_research/outputs/sam2_g1_expansion_20261005/presentation/{r["clip"]}_near_people_reviewed.mp4) | {r["windows"].get("clear",0)} |')
 entries+=['','[数据与质量报告](E:/New_research/reports/sam2_g1_expansion_20261005/RESULTS.md)。新增 2,186 个 mask；严格连续轨迹 21 条 / 82 窗（含静止对照 3 窗）。旧口径新增 180 窗，合计 297 窗；二者不是相同质量门槛。','', '所有标注文件位于各片段 final/；clear_dataset.npz 与 clear_samples.json 是推荐新增子集。当前没有重跑预测网络。']
 (OUT/'presentation/START_HERE.md').write_text('\n'.join(entries)+'\n',encoding='utf8')
 memory='''
Completed: rectified review rejected candidate 17 (escalator crowd); 10 accepted
new intervals / 70 seconds / 700 frames, no source-time overlap with old 53 s.
9 recordings, 5 newly added to the near-person set, old+new union 11 recordings.
Final 67 clip-local IDs, 88 census intervals, 2,186 masks, 50 pending person-
frames and 60 uncertain masks. Strict body-visible subset: 21 continuous runs
>=21 points, 82 stride-5 windows (79 nonstatic + 3 static control), 26 sparse
nonoverlap windows within person runs. Clip 15 yields no strict windows. Same
old status-only rule yields 180 new / 297 combined windows, NOT 297 high-quality
G1 examples. Explicit frame-quality flags and strict arrays are saved. Recordings
and shared person episodes must remain grouped. Offline census, V6 bidirectional
SAM2, donor negatives and explicit repairs all contribute; not automatic tracking
accuracy or independent human GT. Component-mask failure in clip 13 and cropped
persons in clip 15 remain excluded from strict prediction. Final donor-point,
RLE/box/time/projection and all 11 video decode checks passed; 252 original files
unchanged. Old 117 array rows are exact. No forecasting retraining was run.
See `reports/sam2_g1_expansion_20261005/RESULTS.md`, `dataset/clear_dataset.npz`
and `presentation/START_HERE.md` under matching outputs. Existing camera models
trained on newly added recordings require new held-out folds before evaluation.
'''
 ag=ROOT/'AGENTS.md';txt=ag.read_text(encoding='utf8');marker='## Approved six Transformer fusion comparisons';assert marker in txt;txt=txt.replace(marker,memory+'\n'+marker,1);ag.write_text(txt,encoding='utf8')
 index=ROOT/'reports/sam2_tracking_history/INDEX.md';txt=index.read_text(encoding='utf8');txt+='\n\n## G1 新区间近处标注扩充 — 2026 年 10 月 5 日\n\n新增 10 段 / 70 秒，67 个片段内 ID、2,186 个 mask。严格筛选 21 条连续轨迹、82 个窗口（3 个静止对照）；旧口径新增 180 窗，与旧 117 合计 297。没有重跑预测训练。\n\n[报告](E:/New_research/reports/sam2_g1_expansion_20261005/RESULTS.md) · [70 秒完整合集](E:/New_research/outputs/sam2_g1_expansion_20261005/presentation/ALL_10_CLIPS_70s.mp4) · [全部观看入口](E:/New_research/outputs/sam2_g1_expansion_20261005/presentation/START_HERE.md)。旧数据与冻结 V6 核心均保留；这不是新增 V6 版本或精度对照。\n';index.write_text(txt,encoding='utf8')
 regpath=ROOT/'reports/sam2_tracking_history/registry.json';reg=read_json(regpath);reg['updated']='2026-10-05';reg['latest_annotation_dataset']='g1_expansion_20261005';reg['entries'].append(dict(id='g1_expansion_20261005',name='Additional G1 near-person intervals with frozen V6 and explicit offline review',output_directory='outputs/sam2_g1_expansion_20261005',report='reports/sam2_g1_expansion_20261005/RESULTS.md',protocol='outputs/sam2_g1_expansion_20261005/protocol.json',code='scripts/sam2_g1_expansion_20261005/run.py',model='facebook/sam2.1-hiera-large',evaluation_status='Data expansion; 82 strict new windows; not tracking accuracy or human GT; no forecasting retraining',preservation='252 original files verified unchanged; all old assets retained'));write_json(regpath,reg)
 plan=ROOT/'plans/G1_NEAR_EXPANSION_20261005.md';txt=plan.read_text(encoding='utf8');txt+='\n## Executed selection and completion\n\nRectified review excluded candidate 17 (escalator crowd), reducing the initial 11 / 75 s proposal to 10 / 70 s. The final new-recording count is five, excluding 165112. Completed output and evidence are in `reports/sam2_g1_expansion_20261005/RESULTS.md`. No prediction models retrained.\n';plan.write_text(txt,encoding='utf8')
 files=list((ROOT/'scripts/sam2_g1_expansion_20261005').glob('*.py'))+list(report.glob('*.md'))+[OUT/'presentation/START_HERE.md']+list((OUT/'dataset').glob('*'))
 write_json(OUT/'delivery_manifest.json',dict(version='sam2_g1_expansion_20261005',totals=t,files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files if p.is_file()},media=audit['media'],notes='Input, raw masks and intermediate repairs also retained in version directory. Data expansion only.'))
 print('Report, viewing entry, project memory and manifest saved.',flush=True)
if __name__=='__main__':main()
