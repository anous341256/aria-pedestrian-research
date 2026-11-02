"""Persist the observed comparison, uncertainty, source mapping, and case evidence."""
from pathlib import Path
import json,gzip,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001';REPORT=ROOT/'reports/sam2_g1_v6_v8_20261001'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
REPORT.mkdir(exist_ok=True);summary=read_json(OUT/'summary.json')
observations={
'01':'A 两版持续跟随；V8 的 B/C 反复待确认，1.2 秒仍可见却没有输出。V6 的 B 后段出现狭窄局部 mask、随后丢失。',
'02':'A/B 两版较稳定；后段 C/D 发生邻近遮挡。7.9 秒 C 已有可见部分，V6 输出、V8 仍待确认；不把完全遮挡时的空输出直接计作错误。',
'03':'五个抽查时刻四人身份均对应；V8 全段仍有少量待确认空缺，五点抽查未覆盖这些短空缺。',
'04':'V8 在 A/B 清楚可见时提前停止输出；V6 的 A 于末尾 3.9 秒把目标与另一行人合进同一 mask。V8 避免该输出，但未成功恢复目标。',
'05':'单人远距离跟随，两版持续输出，抽查未见明显身份差异。',
'06':'V6 抽查中保持左右两人的身份；V8 的 A 先丢失，3.7 秒 B 的蓝色 mask 落到原 A 身上，末尾两人均无输出。',
'07':'A/B 两版持续跟随；C 为迎面打伞者，V8 在 1.2 秒清楚可见时待确认，随后该人物离开画面。后期无 C 输出不能全算漏跟。',
'08':'近距离双人跟随大体相近；V8 的 B 在最后 4 帧待确认，4.9 秒人物仍清楚可见。',
'09':'A 两版稳定；较远 B/C 有重叠遮挡，V8 的空缺更多，V6 后段局部 mask 也不能当作精确完整标注。',
'10':'A 两版持续跟随；C 原为迎面白 T、黑裤者，离开后两版在 3.7 秒都把 C 重新贴到另一名白上衣行人。可靠性版仍未防住错误重识别。'}
cases=[('01','B',12),('04','A',10),('04','A',20),('04','A',39),('06','B',25),('06','B',37),('08','B',49),('10','C',12),('10','C',25),('10','C',37)]
evidence=[]
for cid,i,f in cases:
    record=dict(clip=cid,id=i,frame=f,clip_seconds=f/10,branches={})
    for v in ['v6','v8']:
        with gzip.open(OUT/cid/v/'decisions.jsonl.gz','rt') as h:rows=[json.loads(l) for l in h]
        b=rows[f]['branches']['motion_with_camera'];d=b['decisions'][i]
        record['branches'][v]=dict(decision=d,candidates=[c for c in b['candidates'] if c['owner']==i])
    evidence.append(record)
write_json(REPORT/'case_evidence.json',evidence)
write_json(REPORT/'visual_review.json',dict(reviewer='assistant visual inspection',method='Reviewed all 10 seed details, 10 five-time contact sheets, and enlarged selected failure frames. Not exhaustive identity/pixel GT.',
    clips=[dict(clip=r['clip'],sample_frames=read_json(OUT/r['clip']/'comparison/stats.json')['sample_frames'],observation=observations[r['clip']]) for r in summary['clips']]))
lines=['# 第 6 版与第 8 版：10 段 G1 候选跟踪对比','',
'**结论：本批测试没有支持第 8 版整体优于第 6 版。第 6 版输出更连续，但会接受合并多人或错误身份；第 8 版能阻止部分异常输出，却产生明显的可见人物漏跟，且仍存在身份转移。**','',
'已完成 10 段 × 2 版，共 20 次完整运行；530 个视频帧、53 秒、29 条初始化人物轨迹。每段 A 是用户筛选的 G1 主目标，额外人物用于观察多人条件下的冲突与误跟。额外人物中包括迎面靠近、遮挡和离开画面的情况，不把全部 29 条都当作纯粹的简单 G1 目标。','',
'[播放 53 秒完整对比](E:/New_research/outputs/sam2_g1_v6_v8_20261001/presentation/all_10_comparison.mp4)｜[展示入口与分段链接](E:/New_research/outputs/sam2_g1_v6_v8_20261001/presentation/START_HERE.md)','',
'## 比较条件','',
'- 第 6 版：原相机运动辅助 V1 的启用旋转分支，运动评分固定 20%。不是运动权重 10% 的第 7 版。',
'- 第 8 版：原可靠性先验组合；保留动态运动可靠度、受限框提示、联合身份证据、隔离待确认和延迟记忆写入。',
'- 两版共用 SAM2.1 Large 权重、2560×1920 图像、10 fps、相同初始 mask/身份和 Aria 旋转参考。每段首帧使用 SAM2 生成种子并目视检查，随后不人工修正、不补标、不按结果重选种子。种子本身是自动分割，不是像素真值。',
'- 两版配置逐项与原版本一致。只适配数据路径、片段长度和从零起算的帧号；原算法文件的哈希一致，适配差异保存在脚本目录的 `.diff` 文件。',
'- 从原始鱼眼导出视频截取所选区间，用设备标定转为线性相机域后再应用旋转单应。姿态与 VRS 时间戳对齐；两版采用同一当前/历史离线 Aria MPS 参考，不使用未来视频帧或未来帧旋转。本轮全部 530 帧均有合格参考姿态，未触发保持上一姿态的回退。',
'- 只处理相机旋转；相机平移、视差和相似衣着仍然存在。未训练新网络。',
'- 模型 revision：`665f8e2ad61cf5f53d65644ff27c8ee525124610`；checkpoint SHA256 见 summary.json。两版均使用同一缺少可选 CUDA 填洞扩展的运行环境。','',
'## 全量输出统计','',
'以下排除每条轨迹的初始化帧。**非空 mask 比例仅反映输出连续性，不是准确率、召回率或身份正确率。** 出画、遮挡、误拒以及真正丢失都可能导致空输出。','',
'| 统计 | 第 6 版 | 第 8 版 |','|---|---:|---:|',
f"| 10 个主目标 A 的非空输出 | 518/520（99.6%） | 455/520（87.5%） |",
f"| 全部 29 条轨迹的非空输出 | 1336/1521（87.8%） | 1110/1521（73.0%） |",
f"| 运动框提示次数 | 1521 | 1163 |",
f"| 自动记忆写入次数 | 1274 | 811 |",'',
'第 8 版少了 226 个人物帧输出。主目标 A 的 63 帧差距全部来自 04、06；其余 8 个主目标，两版都在全部后续帧输出了 mask。这个结论并不保证每个非空 mask 都正确。','',
'## 每段对照','',
'每格为非空 mask 数 / 对应人物帧数，均排除首帧。视频左侧第 6 版，右侧第 8 版，同色同 ID。各段使用共享裁剪、相同缩放；完整画面抽查图另存于各段 comparison/full_*.jpg。','',
'| 片段 | 人数 | A：第6版 / 第8版 | 全部：第6版 / 第8版 | 目视观察 |',
'|---|---:|---|---|---|']
for r in summary['clips']:
    cid=r['clip'];a=r['methods']['v6'];b=r['methods']['v8']
    lines.append(f"| [{cid}](E:/New_research/outputs/sam2_g1_v6_v8_20261001/presentation/{cid}_comparison.mp4) | {r['people']} | {a['main_nonempty']}/{a['main_denominator']} · {b['main_nonempty']}/{b['main_denominator']} | {a['post_seed_nonempty']}/{a['denominator']} · {b['post_seed_nonempty']}/{b['denominator']} | {observations[cid]} |")
lines+=['','## 可以明确定位的失败','',
'### 01-B：正确候选被身份门槛挡住','',
'在片段第 4 帧，V8 候选有效，身份分数约 0.714，低于直接接受门槛 0.72，进入待确认。第 12 帧（原视频 185.2 秒），人物仍可见，原生记忆候选也落在目标附近，但身份分数约 0.556、竞争间隔约 −0.076，未输出。此时运动可靠度降至约 0.348，框提示也因可靠性/尺寸约束停用。这里应区分“候选生成失败”和“候选存在但被门控拒绝”。','',
'### 04-A：第6版接受多人合并，第8版长期漏跟','',
'第 20 帧 V6 仍跟到黑上衣目标，V8 已丢失。第 39 帧（原视频 161.9 秒），V6 原生候选被外观、运动阈值拒绝，但运动框候选覆盖了两个人，SAM 质量分约 0.914、总分约 0.674，超过 0.66 的输出门槛。因此它作为“存疑观察”输出，虽未写 SAM 记忆，错误 mask 仍在最终结果中。V8 此刻没有可用候选；这能避免该错误输出，但并非成功恢复。','',
'### 06-B：内部高可靠性仍然对应错误身份','',
'第 37 帧（原视频 30.7 秒），V6 保持左 B、右 A；V8 已丢失 A，却把 B 的蓝色 mask 贴到右侧 A 身上，并保留左侧小碎片。被选运动框候选的身份分数约 0.814、运动可靠度约 0.998，仍被标为 TRACKED。该帧未写 SAM 记忆，但被接受的观察会进入运动状态更新。**基于自身历史的连续性/残差小，不等于身份正确；相近衣着和错误历史可能使规则自洽。** 这是代码与该帧证据支持的失败解释，尚不是通过单组件消融证明的唯一原因。','',
'### 10-C：两版都出现出画后的错误恢复','',
'C 最初是白 T、黑裤、手持黑包的迎面行人。第 25 帧两版均无 C 输出；第 37 帧（原视频 176.7 秒）两版都把 C 贴到另一名白上衣、浅裤、长发的行人。V6 接受该结果并写入记忆；V8 也接受输出，虽然当帧未写记忆。这说明延迟写入并没有单独解决输出身份错误。','',
'## 证据边界与后续判断','',
'已目视检查十段首帧种子、每段首/四分之一/中点/四分之三/末帧，共 50 个时间点，并放大核查上述关键失败。所有 530 帧均有双版本输出视频和逐帧决策日志。没有完整人工身份与像素真值，不报告 HOTA、IDF1、IoU 准确率，也不能用减少输出宣称“总体误跟率降低”。','',
'这十段来自六份录制，01 与 03 的主目标为同一人，不作为十个独立受试者或独立录制做显著性推断。G1 的主目标筛选不排除邻近人带来的遮挡和身份竞争。','',
'本轮支持保留第 6 版作为连续输出的比较基线，并继续审查第 8 版的误拒与错误恢复。下一步优先区分三个问题：候选是否生成、该候选是否为原身份、是否允许更新运动状态与 SAM 记忆。不要仅继续提高拒绝门槛，也不能把高运动可靠度直接当作身份正确的证据。本轮未修改参数或启动新版本。','',
'## 保存与验证','',
'- [完整统计](E:/New_research/outputs/sam2_g1_v6_v8_20261001/summary.json)',
'- [完整性验证](E:/New_research/outputs/sam2_g1_v6_v8_20261001/verification.json)：20 次运行全部通过；每版 1550 个最终人物帧记录，含空 mask 和种子帧；配置、输入、首帧种子一致；帧号、姿态访问和记忆时序检查通过。',
'- 原版本 11 项与 12 项聚焦检查重新通过，结果保存在本次目录的 v6/tests 与 v8/tests；没有覆盖旧实验。',
'- [目视核查记录](E:/New_research/reports/sam2_g1_v6_v8_20261001/visual_review.json)与[关键帧决策证据](E:/New_research/reports/sam2_g1_v6_v8_20261001/case_evidence.json)。',
'- 每段 input 保存视频、标定、姿态、原帧映射和相同种子；v6/v8 分别保存协议、候选 mask、最终 mask、逐帧判定和运行统计。旧录制、旧标注、旧版本参数与输出均保留。','']
(REPORT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
start=['# 10 段 G1：第 6 版 / 第 8 版','',
'[播放全部 53 秒同步对比](E:/New_research/outputs/sam2_g1_v6_v8_20261001/presentation/all_10_comparison.mp4)','',
'左侧第 6 版，右侧第 8 版。A 是原筛选主目标，同色同身份；彩色区域是最终输出 mask。空白可能是待确认、漏跟、遮挡或出画，不能直接视为正确。','',
'重点先看 04（过度拒绝与多人合并）、06（第8版身份转移）、10（两版错误恢复）。本批没有支持第8版整体优于第6版。','',
'| 片段 | 合集起点 | 单段视频 |','|---|---:|---|']
t=0
for r in summary['clips']:
    cid=r['clip'];start.append(f"| {cid} | {t:.0f} 秒 | [{r['frames']/10:g} 秒完整片段](E:/New_research/outputs/sam2_g1_v6_v8_20261001/presentation/{cid}_comparison.mp4) |");t+=r['frames']/10
start+=['','[逐段统计、失败原因与验证报告](E:/New_research/reports/sam2_g1_v6_v8_20261001/RESULTS.md)','']
(OUT/'presentation/START_HERE.md').write_text('\n'.join(start),encoding='utf8')
assert all(r['pose_held_frames']==0 for r in summary['clips'])
print('Report and review records saved.')
