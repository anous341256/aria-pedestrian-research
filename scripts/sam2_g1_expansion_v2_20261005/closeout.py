"""Audit packaged labels, media, preservation, and write the experiment record."""
from qa import *
import zipfile,xml.etree.ElementTree as ET

def main():
 dest=OUT/'dataset';summary=read_json(dest/'summary.json');t=summary['totals'];package=read_json(dest/'package_audit.json');assert package['all_checks_passed']
 points=0;overlaps=0;events=collections.Counter();unassigned=[];media=[]
 for cid in CLIPS:
  folder=OUT/cid/'final';rows={(r['id'],r['frame']):r for r in v.load(folder/'masks_rle.jsonl.gz')};logs=v.load(folder/'decisions.jsonl.gz')
  for d in logs:
   if (d['id'],d['frame']) not in rows:continue
   for pt in d.get('negative_points',[]):
    donor=rows[pt['donor_id'],pt['frame']];x,y=map(int,pt['xy']);assert v.decode(donor['rle'])[y,x] and pt['frame']==d['frame'] and pt['donor_id']!=d['id'];points+=1
  audit=read_json(OUT/cid/'qa_delivery/audit.json');overlaps+=len(audit['conflicts'])
  review=read_json(folder/'finish_review.json');events.update(e['action'] for e in review['events']);events['final_quarantine']+=len(review['final_quarantine'])
  census=read_json(folder/'census.json')
  unassigned.extend(dict(clip=cid,**x) for x in census['dispositions'] if x['status'].startswith('unassigned'))
  for s in summary['clips']:
   if s['clip']==cid:s['mask_bearing_ids']=len({k[0] for k in rows})
 assert overlaps==0
 for path in sorted((OUT/'presentation').glob('*.mp4')):
  cap=cv2.VideoCapture(str(path));expected=980 if path.name.startswith('ALL') else next(c['frames'] for c in summary['clips'] if path.name.startswith(c['clip']+'_'));count=0
  assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<1e-6
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(1060,1280,3);count+=1
  cap.release();assert count==expected,(path,count,expected);media.append(dict(path=path.name,decoded_frames=count,sha256=digest(path)))
 assert len(media)==11
 old=read_json(OUT/'source_hashes.json');assert all(digest(ROOT/p)==h for p,h in old.items())
 deck=ROOT/'outputs/ppt_midterm_final_20261003/Midterm_Presentation_EN_Final.pptx'
 with zipfile.ZipFile(deck) as z:
  slide=' '.join(ET.fromstring(z.read('ppt/slides/slide13.xml')).itertext())
 assert '162.44' in slide and '238.93' in slide
 audit=dict(final_donor_points_verified=points,final_overlap_alerts=overlaps,events=dict(events),unassigned_proposal_fragments=len(unassigned),original_files_verified=len(old),media=media,ppt_slide13_ade=162.44,ppt_slide13_fde=238.93,all_checks_passed=True)
 write_json(OUT/'delivery_audit.json',audit);write_json(OUT/'unassigned_proposal_fragments.json',unassigned);write_json(dest/'summary.json',summary)
 report=ROOT/'reports'/OUT.name;report.mkdir(parents=True,exist_ok=True)
 table='\n'.join(f"| {c['clip']} | {c['recording'].split('_')[-1]} | {c['seconds']:g} | {c['mask_bearing_ids']} | {c['masks']} | {c['windows'].get('clear',0)} |" for c in summary['clips'])
 text=f'''# 第二轮 G1 近处行人数据扩充

本轮完成 10 段、98 秒、980 帧的新时段标注与连续轨迹筛选。新增 {t['clear_windows']} 个严格筛选窗口，合并后的预测准备数据从 199 增至 {t['combined_strict_windows']} 个窗口。原 199 行全部数组及元数据逐项保留。此轮未训练或重新评估预测网络。

## 数据规模

| 片段 | 录制末尾编号 | 秒 | 有 mask 的片内 ID | mask 数 | 严格窗口 |
|---|---|---:|---:|---:|---:|
{table}

登记 {t['clip_local_ids']} 个片内候选 ID、{t['census_intervals']} 个可见区间，最终保留 {t['masks']} 个 mask；{t['pending']} 个登记人帧处于待核查状态，另有 {t['uncertain']} 个不确定状态。ID 不等于跨视频独立人数，其中包含短时边缘或局部人物。{len(unassigned)} 个未归组的检测片段另存 pending 清单，尚未逐一确认属于重复检测、局部身体还是其他漏标人物，不能据此宣称全员标完。

新增严格子集含 {t['clear_sparse_windows']} 个每条连续区间内间隔 25 帧的稀疏窗口，以及 {t['clear_window_static_controls']} 个静止人物对照窗口。窗口仍可能来自同一人或同一事件，不能把重叠窗口当独立样本。合并预测数据覆盖 {t['union_recordings']} 个录制；新进入预测数据的录制为 {', '.join(t['new_recordings']) or '无'}。

## 标注和筛选

沿用近处优先规则，以图像中人物高度 12% 入选、8% 保留为参考，不限制人物总数，三人一批仅用于调度。先在全部帧发现候选，再根据时序图检查同一人物及别名，使用 V6 与 SAM2 双向离线传播、相同帧已标人物负点，并记录独立提示修正。未用预测误差选样。

抽样时序 mask 图及全部自动异常用于检查身份跳转、只分上衣、背景漂移、边缘裁切和重复。已发现的错误撤回或重新提示；最终仍有大面积 mask 重叠时两侧都列为待核查，并递归撤回依赖失效负点的标签。修正前原始结果及第一轮复核结果均保留。历史 SAM 记忆没有完整重放，因此结果属于经过代理视觉核查的离线自动参考标签，尚非独立人工 GT，也不是纯自动跟踪精度结论。

严格窗口要求 21 个连续有效中心点，对应 6 个历史点和 15 个未来点，步长 0.1 秒，窗口步幅 5 帧。排除缺失、不确定、显著遮挡、身份不清、身体片段、图像边缘裁切以及 mask 与指引框明显不一致的帧。不插值 mask 或中心来补足窗口。短间隔指引框插值仅用于提示，来源在日志中保留。

## 场景和后续划分

这批覆盖白天人行道、站台、站厅、商场通道、有顶步行区及夜间街道。22、24、26、29、30 保留 G1 边界标记：目标交互较少，但仍有较大头部旋转、遮挡或裁切，不能把整批称为纯 G1。静止人物按人物标记，不按整段视频推断。新的源时间区间与前两批不重叠；同录制及共享人物事件必须一起划分。

已有全数据相机训练包含新录制。下一轮预测评估需要匹配新的录制排除组合，不能直接用见过测试录制的相机模型。新增数据还没有新的 ADE 结果。

## PPT 方案的 ADE

最终英文 PPT 第 10 页的方法图和第 13 页结果对应独立相机分支与行人分支、带历史重建的模型。它在原 117 个窗口、6 个录制划分、3 个种子上的 ADE 为 **162.44 像素**，FDE 为 **238.93 像素**。保存的旧耦合基线为 158.06 / 257.34 像素；保持最后位置为 141.43 / 218.35 像素。PPT 中 73.92 像素是替换为参考未来旋转的诊断，不是正常推理 ADE。这些均为已完成旧实验的结果。

## 文件与验证

视频入口为 `presentation/ALL_10_CLIPS_98s.mp4`，另有 10 段单独视频和总览图。最终标签位于各片段 `final/`。推荐下一轮读取 `dataset/combined_strict_dataset.npz` 与 `combined_strict_samples.json`；新增独立子集为 `clear_dataset.npz` 与 `clear_samples.json`。兼容宽松窗口仅留作检查，不作为推荐训练集。

全部 11 个视频逐帧解码通过；mask 编码、外接框、像素数、相机时间对齐和窗口连续性检查通过。原 {len(old)} 个受保护文件哈希未变，旧 199 行数组及元数据完全一致；最终 {points} 个保留标签负点均落在对应同帧供体 mask 内，最终大面积重叠报警为零。这些是工程一致性检查，不代表零标注误差。
'''
 (report/'RESULTS.md').write_text(text,encoding='utf8')
 (OUT/'presentation/START_HERE.md').write_text(f'''# 第二轮 G1 扩充标注预览

先看 [98 秒合集](ALL_10_CLIPS_98s.mp4)，再按编号查看单段视频。partial / review 表示该帧不直接进入清晰连续轨迹子集；没有 mask 的待核查帧未补画。

新增 {t['clear_windows']} 个严格窗口，合并后 {t['combined_strict_windows']} 个。详细记录：[扩充报告](../../../reports/{OUT.name}/RESULTS.md)。
''',encoding='utf8')
 print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
