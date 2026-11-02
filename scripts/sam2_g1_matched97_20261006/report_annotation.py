"""Finalize annotation inventory, source checks, and reproducible dataset facts."""
from qa import *
from reviewed_people import REJECTED_CLIPS
def main():
 dest=OUT/'dataset';summary=read_json(dest/'summary.json');meta=read_json(dest/'clear_samples.json');old=read_json(ROOT/'outputs/transformer_expanded_20261005/data/samples.json')
 oldrecs={m['recording'] for m in old};newrecs={m['recording'] for m in meta}
 summary['totals'].update(new_recordings=sorted(newrecs-oldrecs),union_recordings=len(oldrecs|newrecs));write_json(dest/'summary.json',summary)
 audits=[read_json(OUT/c/'qa_delivery/audit.json') for c in CLIPS];assert sum(len(a['conflicts']) for a in audits)==0
 sources=read_json(dest/'sources.json');assert all(digest(ROOT/p)==h for p,h in sources.items())
 windows=[]
 for m in meta:
  assert len(m['frames'])==21
  windows.append((m['clip'],m['id'],m['cutoff']))
 assert len(set(windows))==97
 clips=collections.Counter(m['clip'] for m in meta);tracks={(m['clip'],m['id']) for m in meta};t=summary['totals']
 checks=dict(passed=True,selected_windows=97,unique_clip_local_ids=len(tracks),final_pair_overlaps=0,final_sources_verified=len(sources),g1_screen='20 deg/1s; p95 50 deg/s',static=16,moving=81)
 write_json(dest/'delivery_audit.json',checks)
 report=ROOT/'reports/sam2_g1_matched97_20261006';report.mkdir(parents=True,exist_ok=True)
 lines=['# 新 G1 等量标注批次','',f"本次筛查23个新时段，完成20段、147秒、1470帧的近处人物标注尝试；100个片段内ID、4037个保留mask。经过连续身体、身份、遮挡与相机运动筛选，共99个可用窗口，冻结97个用于批次替换对照，涉及15段视频、9次录制、{len(tracks)}条片段内轨迹。不是97个独立人，也不是100个全局不同的人。",'',
 '每个窗口为过去6点和未来15点，采样0.1秒、窗口步幅5帧。新池为81个行走窗口和16个静止窗口；旧池87/10。实际训练按每轮可用窗口分别匹配行走和静止数量。','',
 '筛选条件：沿用近处人物优先，表观框高达到12%入场、低至8%仍可维持；标注完整身体且不触及图像边缘，mask与引导框一致性检查、明确遮挡排除、连续21帧、相机姿态有效及时间对齐。逐窗口最大1秒参考旋转≤20°、角速度95分位≤50°/秒是本次本地G1规则，不是论文的分类阈值。整段视频不一定均属于G1。','',
 'SAM2.1 large + V6双向离线传播、分批负点、锚点及逐帧修正保持原方法。视觉抽查与显式排除参与标注，未独立人工逐像素重标。保留所有失败、部分身体和待核查条目，不用插值补造mask中心。','',
 '## 入选数量','', '| 片段 | 入选窗口 | 场景 |','|---|---:|---|']
 scene={f'{r["candidate"]:02d}':r['visual_observation'] for r in read_json(OUT/'selected_clips.json')}
 for cid,n in sorted(clips.items()):lines.append(f'| {cid} | {n} | {scene[cid]} |')
 lines+=['','## 核查','',
 f"最终4037个保留mask不存在超过规则阈值的成对重叠；仍有{t['pending']}个pending、{t['uncertain']}个uncertain人帧。严格窗口排除这些状态。零重叠告警不等于零身份错误。",
 '', '原始数据和旧模型文件哈希核查通过，新时段与此前原始10段、第一批及第二批扩充时段无时间重叠。新数据没有直接加入旧296形成393；实际对照使用同一基础199和两个可替换的97窗口池。', '',
 '## 浏览入口','',
 '[实际入选窗口合集：15例，63秒](../../outputs/sam2_g1_matched97_20261006/presentation/SELECTED_G1_WINDOWS.mp4)',
 '', '[全部标注尝试合集：20段，147秒](../../outputs/sam2_g1_matched97_20261006/presentation/ALL_REVIEWED_CLIPS.mp4)',
 '', '展示窗口按各片段入选列表时间中间位置确定，没有根据预测结果挑选；每例实际2.1秒播放两次。所有单段视频、逐帧mask、修正记录和未入选原因均保留。']
 (report/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(checks,flush=True)
if __name__=='__main__':main()
