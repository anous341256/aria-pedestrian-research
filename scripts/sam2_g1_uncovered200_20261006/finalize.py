"""Aggregate all selected windows, verify provenance, and publish review index."""
from audit import *
import argparse

PRESENT=OUT/'presentation/review_v2'
REPORT=ROOT/'reports/sam2_g1_uncovered200_20261006'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify():
 result={}
 for name,base in [('selection_hashes.json',OUT),('protected_previous_hashes.json',ROOT)]:
  hashes=read(OUT/name)
  for p,h in hashes.items():assert sha(base/p)==h,('changed',str(base/p))
  result[name]=dict(checked=len(hashes),unchanged=True)
 old=read(OUT/'previous_coverage.json');windows=read(OUT/'selected_windows.json');clips=read(OUT/'selected_clips.json')
 assert len(windows)==200 and len({w['key'] for w in windows})==200
 pool={w['key'] for w in read(OUT/'available500.json')}
 clipmap={c['clip']:c for c in clips};window_frames=set();clip_frames=set()
 for w in windows:
  assert w['key'] in pool
  assert not any(w['recording']==o['recording'] and w['start_frame']<o['end_frame_exclusive'] and w['start_frame']+21>o['start_frame'] for o in old)
  c=clipmap[w['clip']];assert w['local_start_frame']==w['start_frame']-c['source_start_frame']
  assert w['local_cutoff_frame']==w['local_start_frame']+5
  assert w['recording']==c['recording_group'] and w['start_frame']+21<=c['source_end_frame_exclusive']
  window_frames.update((w['recording'],f) for f in range(w['start_frame'],w['start_frame']+21))
 for c in clips:clip_frames.update((c['recording_group'],f) for f in range(c['source_start_frame'],c['source_end_frame_exclusive']))
 assert window_frames==clip_frames and len(window_frames)==3204
 poses=0;maxgap=0;videos=[]
 for c in clips:
  cid=c['clip'];inp=OUT/cid/'input';meta=read(inp/'metadata.json');align=read(inp/'alignment.json');z=np.load(inp/'reference.npz');n=meta['frames']
  assert len(align)==len(z['R'])==len(z['times_ns'])==n==c['source_end_frame_exclusive']-c['source_start_frame']
  assert [a['source_frame'] for a in align]==list(range(c['source_start_frame'],c['source_end_frame_exclusive']))
  assert all(a['status']=='valid' and a['quality']>=.5 and a['nearest_gap_ms']<=30 for a in align)
  assert len({a['graph'] for a in align})==1
  assert np.all((np.diff(z['times_ns'])/1e9>=.08)&(np.diff(z['times_ns'])/1e9<=.12))
  assert all(np.isfinite(z[k]).all() for k in z.files)
  assert np.max(np.abs(z['R'].transpose(0,2,1)@z['R']-np.eye(3)))<1e-5
  for p,h in meta['hashes'].items():assert sha(inp/p)==h,(cid,p)
  assert meta['pose_held_frames']==0
  poses+=n;maxgap=max(maxgap,max(a['nearest_gap_ms'] for a in align))
  media=read(PRESENT/'clips'/f'{cid}_media.json');videos.append((Path(media['video']),n))
 timeline=read(PRESENT/'timeline.json');total=sum(c['source_end_frame_exclusive']-c['source_start_frame'] for c in clips)
 videos.append((PRESENT/'ALL_114_CLIPS.mp4',total))
 videos.append((PRESENT/'ALL_WITH_MASKS.mp4',sum(r['frames'] for r in timeline if r['with_masks_start_s'] is not None)))
 for group in sorted({r['group'] for r in timeline}):
  rs=[r for r in timeline if r['group']==group];videos.append((Path(rs[0]['group_video']),sum(r['frames'] for r in rs)))
 for p,n in videos:
  cap=cv2.VideoCapture(str(p));count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape==(1080,1280,3),(p,frame.shape)
   count+=1
  fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==n and abs(fps-10)<1e-6,(p,count,n,fps)
 result.update(selected_windows=200,selected_clips=len(clips),source_overlap_with_previous63=0,valid_pose_frames=poses,max_nearest_pose_gap_ms=maxgap,pose_held_frames=0,input_hashes_checked=len(clips)*3,decoded_review_videos=len(videos),unique_frames=total)
 write(OUT/'verification.json',result);return result

def aggregate():
 clips=read(OUT/'selected_clips.json');windows=read(OUT/'selected_windows.json');ledger=[];cl=[];runs=[];person_windows=[]
 for c in clips:
  cid=c['clip'];folder=OUT/cid;a=read(folder/'review_v2/audit.json');census=read(folder/'census.json');rows=v.load(folder/'review_v2/masks_rle.jsonl.gz');raw=v.load(folder/'masks_rle.jsonl.gz')
  keys={(r['id'],r['frame']) for r in rows};rawkeys={(r['id'],r['frame']) for r in raw};depkeys={(r['id'],r['frame']) for r in a['donor_dependencies']}
  ids=sorted({r['id'] for r in rows})
  for pid in ids:
   fs=sorted(f for i,f in keys if i==pid);start=last=fs[0]
   for f in fs[1:]+[10**9]:
    if f!=last+1:
     runs.append(dict(clip=cid,id=pid,start_frame=start,end_frame=last,frames=last-start+1))
     start=f
    last=f
  for w in [w for w in windows if w['clip']==cid]:
   lo=w['local_start_frame'];hi=lo+20;wk={(i,f) for i,f in keys if lo<=f<=hi};wraw={(i,f) for i,f in rawkeys if lo<=f<=hi};wi=sorted({i for i,f in wk});complete=[i for i in wi if all((i,f) in wk for f in range(lo,hi+1))];clean=[i for i in complete if not any((i,f) in depkeys for f in range(lo,hi+1))]
   if a['scene_exclusion']:status='scene_or_target_excluded'
   elif wk:status='provisional_masks_available'
   elif wraw:status='raw_masks_all_pending'
   elif not census['segments']:status='no_retained_near_proposal'
   else:status='no_mask_in_window'
   item=dict(window_id=w['window_id'],key=w['key'],clip=cid,recording=w['recording'],source_start_s=w['start_s'],source_end_s=w['end_s'],local_start_frame=lo,local_end_frame=hi,inclusion_probability=w['inclusion_probability'],status=status,scene_exclusion=a['scene_exclusion'],raw_masks=len(wraw),review_masks=len(wk),ids=wi,complete21_ids=complete,complete21_without_donor_flag_ids=clean)
   ledger.append(item)
   person_windows.extend(dict(window_id=w['window_id'],clip=cid,id=i,local_start_frame=lo,local_end_frame=hi,donor_dependency_flag=i not in clean,status='continuous_mask_candidate_only_user_review_pending') for i in complete)
  cl.append(dict(clip=cid,recording=c['recording_group'],source_start_s=c['start_s'],source_end_s=c['end_s'],frames=a['frames'],window_ids=c['window_ids'],raw_ids=a['raw_ids'],review_ids=a['review_ids'],raw_masks=a['raw_masks'],review_masks=a['review_masks'],removed_raw_masks=len(rawkeys-keys),expected_person_frames=a['expected_person_frames'],missing_person_frames=a['missing_person_frames'],pending_records=a['pending_masks'],overlap_pair_frames=len(a['conflicts']),donor_dependency_person_frames=len(depkeys),scene_exclusion=a['scene_exclusion'],video=str(PRESENT/'clips'/f'{cid}_masks.mp4')))
 ledger.sort(key=lambda r:r['window_id']);assert len(ledger)==200
 post=read(OUT/'post_review.json')
 summary=dict(selected_windows=len(ledger),recordings=len({w['recording'] for w in windows}),merged_clips=len(cl),unique_frames=sum(r['frames'] for r in cl),unique_seconds=sum(r['frames'] for r in cl)/10,window_status_counts=dict(collections.Counter(w['status'] for w in ledger)),clips_with_review_masks=sum(r['review_masks']>0 for r in cl),scene_or_target_excluded_clips=sum(bool(r['scene_exclusion']) for r in cl),raw_masks=sum(r['raw_masks'] for r in cl),review_masks=sum(r['review_masks'] for r in cl),raw_clip_local_ids=sum(r['raw_ids'] for r in cl),review_clip_local_ids=sum(r['review_ids'] for r in cl),removed_raw_masks=sum(r['removed_raw_masks'] for r in cl),missing_person_frames=sum(r['missing_person_frames'] for r in cl),pending_records=sum(r['pending_records'] for r in cl),overlap_pair_frames=sum(r['overlap_pair_frames'] for r in cl),donor_dependency_person_frames=sum(r['donor_dependency_person_frames'] for r in cl),continuous_mask_runs=len(runs),runs_at_least21_frames=sum(r['frames']>=21 for r in runs),complete21_person_windows=len(person_windows),windows_with_complete21_masks=sum(bool(w['complete21_ids']) for w in ledger),complete21_person_windows_without_donor_flag=sum(not r['donor_dependency_flag'] for r in person_windows),temporal_proposal_review_pages=len(post['reviewed_track_pages']),explicit_temporal_proposal_rejections=sum(len(r) for r in post['reject_proposals'].values()),status='Provisional annotation expansion only; user review pending; no training merge')
 for name,data in [('window_review_ledger.json',ledger),('clip_review_ledger.json',cl),('continuous_mask_runs.json',runs),('continuous21_person_window_candidates.json',person_windows),('summary.json',summary)]:write(OUT/name,data)
 return summary,ledger,cl

def link(p,label):return f'[{label}](<{Path(p).as_posix()}>)'
def mm(s):return f'{int(s)//60:02d}:{int(s)%60:02d}.{int(round(s*10))%10}'
def reports(s,windows,clips):
 REPORT.mkdir(parents=True,exist_ok=True);timeline=read(PRESENT/'timeline.json');by={r['clip']:r for r in timeline}
 lines=['# 200 个未覆盖窗口：标注检查入口','',f"已处理固定抽样的 **200 个时间窗口、25 个录制**，合并为 **114 段、320.4 秒**。共有 **{s['window_status_counts'].get('provisional_masks_available',0)} 个窗口**保留初筛 mask；这些仍需你检查。200 是抽样窗口数，不是已通过质量验收的人物轨迹数。",'',link(PRESENT/'ALL_114_CLIPS.mp4','播放全部 114 段（5 分 20.4 秒）'),'','## 分组查看','']
 lines.insert(4,link(PRESENT/'ALL_WITH_MASKS.mp4','优先播放：全部有初筛 mask 的片段合集'))
 lines.insert(5,'')
 for group in range(1,5):
  rs=[r for r in timeline if r['group']==group];label=f"第 {group} 组：片段 {rs[0]['clip']}–{rs[-1]['clip']}"
  lines.append(f"- {link(rs[0]['group_video'],label)}（{sum(r['frames'] for r in rs)/10:.1f} 秒）")
 lines+=['','视频标题显示片段编号，人物旁显示 Nxx。请按“片段编号 + 人物编号 + 大约时间”指出漏标、串人或 mask 不完整；编号仅在当前片段内有效。视频底部是对应的 Wxxx 窗口。','',f"初筛保留 {s['review_masks']:,} 个 mask、{s['review_clip_local_ids']} 个片段内人物 ID；其中 {s['complete21_person_windows']} 个“人物×窗口”有连续 21 帧 mask，分布在 {s['windows_with_complete21_masks']} 个时间窗口。这只是连续性候选，身份和边界质量尚未由你确认。",'','没有 mask 的片段也保留在合集里，避免把抽中的空场景或不适合 G1 的片段悄悄替换掉。灰区、遮挡、重复身份、检测失败等原因见完整索引和报告。','',link(REPORT/'RESULTS.md','完整统计、方法与限制'),'','## 逐段索引','', '| 片段 | 窗口 | 初筛 ID / mask | 全集位置 | 单独视频 | 说明 |','|---|---|---:|---|---|---|']
 for c in clips:
  t=by[c['clip']];reason='场景/目标暂不纳入' if c['scene_exclusion'] else ('待检查' if c['review_masks'] else '暂无保留 mask')
  lines.append(f"| {c['clip']} | {', '.join(c['window_ids'])} | {c['review_ids']} / {c['review_masks']} | {mm(t['all_start_s'])} | {link(c['video'],'播放')} | {reason} |")
 (OUT/'presentation/START_HERE.md').write_text('\n'.join(lines)+'\n','utf8')
 statuscn={'provisional_masks_available':'保留至少一个初筛 mask','scene_or_target_excluded':'RGB 检查后场景或目标不符合当前范围','raw_masks_all_pending':'原始 mask 全部被列为待核查','no_retained_near_proposal':'无保留的近人候选','no_mask_in_window':'该窗口内未生成 mask'}
 report=['# G1 未覆盖 200 窗口标注扩充（2026-10-06）','',link(OUT/'presentation/START_HERE.md','检查入口'),'','## 本次结果','',f"固定处理 200 个时间窗口，来自 25 个录制；重叠窗口合并为 114 段 / 3,204 帧 / 320.4 秒。选中帧与既有 63 段 G1 近人标注区间零重叠。这个未覆盖定义针对既有 G1 近人数据清单，不是宣称从未进行过任何历史探索性 SAM 尝试。",'','| 窗口处理结果 | 数量 |','|---|---:|']
 for k,n in s['window_status_counts'].items():report.append(f'| {statuscn[k]} | {n} |')
 report+=['',f"原始 SAM 输出 {s['raw_masks']:,} 个 mask / {s['raw_clip_local_ids']} 个片段内 ID。初筛展示 {s['review_masks']:,} 个 mask / {s['review_clip_local_ids']} 个片段内 ID，涉及 {s['clips_with_review_masks']} 段。ID 可能仍有重复，不能当作独立真实人数。",'',f"共有 {s['complete21_person_windows']} 个连续 21 帧人物窗口候选，涉及 {s['windows_with_complete21_masks']} 个时间窗口；其中 {s['complete21_person_windows_without_donor_flag']} 个候选没有下述负点来源标记。连续 mask 不证明身份正确或可直接用于训练，本次没有合并训练集。",'',f"有 {s['scene_or_target_excluded_clips']} 段因上下楼/扶梯、上下车或车厢、仅手臂/屏幕等场景或目标问题暂不纳入。原始选样和 RGB 视频仍保留。相机条件合格并不保证属于当前平地行走 G1 目标场景。",'','## 抽样与覆盖','', '从 500 个完全未覆盖且相机条件合格的候选中，在查看新 RGB/人物之前，按录制和录制内候选时间位置（最多四段）分为 98 层。固定种子 2026100623；每层先取一例，其余名额按剩余容量比例分配，再在层内无放回随机抽样。没有因无人、困难或失败而替换样本。各窗口抽中概率记录在清单中；这是分层抽样，不是所有窗口等概率，后续总体分布统计应使用概率权重。','', '每窗口为 10 Hz 的 21 帧，历史 6 点 / 未来 15 点，点间跨度 2.0 秒、文件帧覆盖 2.1 秒。相互重叠或相接的窗口合并跟踪，不跨未选时间间隙。标注使用整个片段，未来帧可用于离线标注；未来 RGB/参考旋转不因此成为预测模型输入。','', '## 标注流程与检查','', '校准针孔 RGB + Aria 离线参考旋转 → 全帧和分块人物发现 → 显示尺度筛选及时间关联 → RGB 候选条带检查 → V6 / SAM2.1 large 双向传播 → mask 数值/重叠筛查 → 编号视频。近人的入选高度阈值为图高 12%，保留阈值为 8%，并非米制距离；不设总人数上限，每批最多三个仅用于调度。短暂进入/离开画面的目标保留，但可能不能构成预测窗口。','', '沿用 V6 候选评分与记忆处理，已有批次的可靠当前 mask 可作为后续批次负点。RGB 检查显式剔除了海报、反射、身体局部重复以及明显串人的 proposal。先处理片段的原始 SAM 已保留，筛查在输出层执行；后处理片段在 SAM 前应用相同显式拒绝记录。因此这是有检查辅助的离线标注流程，不是全自动准确率对照实验。', '',f"所有 114 段做过场景缩略图检查，{s['temporal_proposal_review_pages']} 页候选时间条带做过查看，另有 {s['explicit_temporal_proposal_rejections']} 条显式 proposal 拒绝记录。实际 SAM mask 的抽样检查见 mask_visual_review.json。没有逐帧人工真值验收。",'', '自动初筛要求 mask 与提示框 IoU ≥ 0.45，框外比例 ≤ 0.10，非 V6_UNCERTAIN，mask 高度 ≥ 图高 8%。尚合格的两个人 mask 若交集超过较小 mask 一半，双方先挂起；不靠强制裁剪伪造身份正确性。', '',f"原始输出中 {s['removed_raw_masks']} 个 mask 未进入展示；另有 {s['missing_person_frames']} 个预期人物帧没有原始 mask。pending.json 共 {s['pending_records']} 条记录，包含显式拒绝区间的缺失帧，不能直接当作被删除 mask 数。未解决的重叠冲突共 {s['overlap_pair_frames']} 个身份对×帧。",'',f"保留 mask 中 {s['donor_dependency_person_frames']} 个人物帧使用过后来没有进入展示集合的原始负点供体。该依赖已标记，但没有重跑记忆，因此不能据此假定完全不受早期错误影响。",'','## 交付与边界','', '- 原始候选：每段根目录 masks_rle.jsonl.gz、decisions.jsonl.gz、census.json 和 segments/。','- 初筛标签：每段 review_v2/；review_v1/ 和旧 review_candidates/ 均为早期草稿，不作为本次交付。','- 展示：presentation/review_v2/，114 段单独视频、全集、四组视频、29 页缩略图。','- window_review_ledger.json：固定 200 窗口逐个可追溯的处理状态。','- continuous21_person_window_candidates.json：有连续 mask 的人物窗口候选，不是训练集。','- verification.json：锁定选择、旧资产哈希、时序与几何有效性、视频完整解码核对。','', '尚需用户检查身份连续、整个人是否完整、遮挡边界、漏检和重复 ID。本批未运行预测训练，未并入旧数据，未上传 CVAT。旧数据与模型保留。']
 (REPORT/'RESULTS.md').write_text('\n'.join(report)+'\n','utf8')
 with (REPORT/'RESULTS.md').open('a',encoding='utf8') as f:
  visual=read(OUT/'mask_visual_review.json')
  f.write(f"\n## 本次人工式检查记录\n\n助手已查看 {len(visual['reviewed_contact_pages'])} 页实际 mask 的首/中/末帧缩略图；这与用户逐帧验收不同。另对 010 的 19 条候选及 100 的 10 条候选查看了五页时间条带，撤回两处过宽的场景排除，分别补标八个和四个候选身份。100 的 P1 在第 10 帧截断，保留串人之前的短轨迹。015/N05、017/N01、021/N03、050/N03 因背景误分割或身体不完整整条挂起；原始 mask 未删除，V2 重新执行重叠和负点来源筛查。\n\nSAM2 运行环境的可选 CUDA 孔洞后处理扩展不可用，日志明确记录跳过该步骤；本批未更改环境或用其他方法静默填补。\n")

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
 if a.verify:print(json.dumps(verify(),ensure_ascii=False),flush=True)
 s,w,c=aggregate();reports(s,w,c);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
