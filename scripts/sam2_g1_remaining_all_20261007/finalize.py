"""Account for the complete locked remainder and build a separate screened candidate pack."""
from audit import *
import argparse,html
PRESENT=OUT/'presentation/review_v2'
REPORT=ROOT/'reports'/OUT.name

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def aggregate():
 clips=read(OUT/'selected_clips.json');windows=read(OUT/'selected_windows.json');visual=read(OUT/'mask_visual_review.json')
 ledger=[];cl=[];candidates=[];strict=[];runs=[];arrays=[]
 for c in clips:
  cid=c['clip'];folder=OUT/cid;a=read(folder/'review_v2/audit.json');census=read(folder/'census.json');rows=v.load(folder/'review_v2/masks_rle.jsonl.gz');raw=v.load(folder/'masks_rle.jsonl.gz');ref=np.load(folder/'input/reference.npz')
  by={(r['id'],r['frame']):r for r in rows};rawkeys={(r['id'],r['frame']) for r in raw};depkeys={(r['id'],r['frame']) for r in a['donor_dependencies']}
  for pid in sorted({r['id'] for r in rows}):
   fs=sorted(f for i,f in by if i==pid);start=last=fs[0]
   for f in fs[1:]+[10**9]:
    if f!=last+1:runs.append(dict(clip=cid,id=pid,start_frame=start,end_frame=last,frames=last-start+1));start=f
    last=f
  for w in (w for w in windows if w['clip']==cid):
   lo=w['local_start_frame'];hi=lo+20;wk={(i,f) for i,f in by if lo<=f<=hi};wraw={(i,f) for i,f in rawkeys if lo<=f<=hi};wi=sorted({i for i,f in wk});complete=[i for i in wi if all((i,f) in by for f in range(lo,hi+1))];accepted=[]
   for pid in complete:
    reasons=[];segs=[e for e in census['segments'] if e['id']==pid and e['start']<=lo and e['end']>=hi]
    if len(segs)!=1:reasons.append('not_one_continuous_census_segment')
    if segs and any((pid,f) in depkeys for f in range(segs[0]['start'],segs[0]['end']+1)):reasons.append('segment_has_rejected_negative_donor_dependency')
    rs=[by[pid,f] for f in range(lo,hi+1)];boxes=np.array([r['box'] for r in rs],float)
    if np.any((boxes[:,0]<=5)|(boxes[:,1]<=5)|(boxes[:,2]>=2555)|(boxes[:,3]>=1915)):reasons.append('border_truncation')
    if np.any((boxes[:,3]-boxes[:,1])<.08*1920):reasons.append('below_8pct_height')
    reject=visual['strict_exclude_ids'].get(cid,{}).get(pid)
    if reject:reasons.append('visual_body_quality: '+reject)
    for cut in visual.get('strict_exclude_ranges',{}).get(cid,{}).get(pid,[]):
     if lo<=cut['end'] and hi>=cut['start']:reasons.append('visual_range_quality: '+cut['reason'])
    if a['scene_exclusion']:reasons.append('scene_excluded')
    item=dict(key=f'{cid}_{pid}_{lo:03d}',window_id=w['window_id'],clip=cid,id=pid,segment=segs[0]['segment'] if len(segs)==1 else None,recording=w['recording'],source_start_frame=w['start_frame'],local_start_frame=lo,local_end_frame=hi,coverage_status=w['coverage_status'],exclusion_reasons=reasons,eligible=not reasons,status='Screened offline SAM reference; user review pending; not independently labelled GT')
    candidates.append(item)
    if not reasons:
     strict.append(item);accepted.append(pid);arrays.append(dict(xy=(boxes[:,:2]+boxes[:,2:])/2,boxes=boxes,R=ref['R'][lo:hi+1],K=ref['K'],position=ref['position'][lo:hi+1],times_ns=ref['times_ns'][lo:hi+1]))
   status='scene_or_target_excluded' if a['scene_exclusion'] else ('provisional_masks_available' if wk else ('raw_masks_all_pending' if wraw else ('no_retained_near_proposal' if not census['segments'] else 'no_mask_in_window')))
   ledger.append(dict(window_id=w['window_id'],key=w['key'],clip=cid,recording=w['recording'],source_start_s=w['start_s'],source_end_s=w['end_s'],local_start_frame=lo,status=status,scene_exclusion=a['scene_exclusion'],raw_masks=len(wraw),review_masks=len(wk),complete21_ids=complete,strict_candidate_ids=accepted,coverage_status=w['coverage_status']))
  cl.append(dict(clip=cid,recording=c['recording_group'],frames=a['frames'],window_ids=c['window_ids'],raw_masks=a['raw_masks'],review_masks=a['review_masks'],raw_ids=a['raw_ids'],review_ids=a['review_ids'],missing_person_frames=a['missing_person_frames'],pending_masks=a['pending_masks'],donor_dependency_person_frames=len(depkeys),scene_exclusion=a['scene_exclusion'],video=str(PRESENT/'clips'/f'{cid}_masks.mp4')))
 ledger.sort(key=lambda r:r['window_id']);assert len(ledger)==225
 def sparse(key):
  last={};keep=[]
  for r in sorted(strict,key=lambda x:(x['recording'],x['source_start_frame'],x['key'])):
   k=key(r)
   if r['source_start_frame']>=last.get(k,-1):keep.append(r['key']);last[k]=r['source_start_frame']+21
  return keep
 sparse_person=sparse(lambda r:(r['clip'],r['id']));sparse_time=sparse(lambda r:r['recording'])
 s=dict(selected_windows=len(ledger),clips=len(cl),recordings=len({c['recording'] for c in cl}),frames=sum(c['frames'] for c in cl),new_source_frames=1813,old_context_frames=1305,window_status_counts=dict(collections.Counter(w['status'] for w in ledger)),raw_masks=sum(c['raw_masks'] for c in cl),review_masks=sum(c['review_masks'] for c in cl),raw_clip_local_ids=sum(c['raw_ids'] for c in cl),review_clip_local_ids=sum(c['review_ids'] for c in cl),clips_with_masks=sum(c['review_masks']>0 for c in cl),scene_excluded_clips=sum(bool(c['scene_exclusion']) for c in cl),missing_person_frames=sum(c['missing_person_frames'] for c in cl),donor_dependency_person_frames=sum(c['donor_dependency_person_frames'] for c in cl),complete21_person_windows=len(candidates),strict_person_windows=len(strict),strict_time_windows=len({r['window_id'] for r in strict}),strict_clips=len({r['clip'] for r in strict}),strict_recordings=len({r['recording'] for r in strict}),strict_clip_local_ids=len({(r['clip'],r['id']) for r in strict}),within_identity_nonoverlap=len(sparse_person),recording_time_nonoverlap=len(sparse_time),strict_exclusion_reason_counts=dict(collections.Counter(reason for r in candidates for reason in r['exclusion_reasons'])),continuous_mask_runs=len(runs),runs_at_least21=sum(r['frames']>=21 for r in runs),mask_review_pages=len(visual['reviewed_contact_pages']),quarantined_ids=sum(len(x) for x in visual['quarantine_ids'].values()),training_or_active_merge=False)
 s['strict_coverage_status_counts']=dict(collections.Counter(r['coverage_status'] for r in strict))
 s['strict_recording_counts']=dict(collections.Counter(r['recording'] for r in strict))
 s['visual_quality_excluded_candidates']=sum(any(x.startswith('visual_') for x in r['exclusion_reasons']) for r in candidates)
 for name,data in [('summary.json',s),('window_review_ledger.json',ledger),('clip_review_ledger.json',cl),('continuous21_person_window_candidates.json',candidates),('strict_person_window_candidates.json',strict),('continuous_mask_runs.json',runs),('sparse_candidates.json',dict(within_clip_identity=sparse_person,recording_time=sparse_time))]:write(OUT/name,data)
 dest=OUT/'dataset';dest.mkdir(exist_ok=True)
 if arrays:np.savez_compressed(dest/'strict_candidates.npz',**{k:np.stack([r[k] for r in arrays]) for k in arrays[0]},keys=np.array([r['key'] for r in strict]),recordings=np.array([r['recording'] for r in strict]))
 write(dest/'metadata.json',strict)
 return s,ledger,cl

def verify():
 result={}
 for name,base in [('selection_hashes.json',OUT),('protected_previous_hashes.json',ROOT)]:
  hashes=read(OUT/name)
  for p,h in hashes.items():assert sha(base/p)==h,('changed',str(base/p))
  result[name]=dict(checked=len(hashes),unchanged=True)
 clips=read(OUT/'selected_clips.json');windows=read(OUT/'selected_windows.json');old=read(OUT/'previous_coverage.json');locked=read(OUT/'selection_locked.json');coverage=read(OUT/'coverage_ledger.json');visual=read(OUT/'mask_visual_review.json')
 assert len(coverage)==835 and len(windows)==225 and len({w['key'] for w in windows})==225
 assert {r['key'] for r in coverage if r['status'].startswith('remaining')}=={w['key'] for w in windows}
 clipmap={c['clip']:c for c in clips};frames=set();poses=0;maxgap=0;videos=[];hasheschecked=0
 for w in windows:
  c=clipmap[w['clip']];assert w['local_start_frame']==w['start_frame']-c['source_start_frame'];assert w['start_frame']+21<=c['source_end_frame_exclusive'];assert w['recording']==c['recording_group']
  assert not any(o.get('user_rejected') and o['recording']==w['recording'] and o['start_frame']<w['start_frame']+21 and o['end_frame_exclusive']>w['start_frame'] for o in old)
 for c in clips:
  cid=c['clip'];inp=OUT/cid/'input';meta=read(inp/'metadata.json');align=read(inp/'alignment.json');z=np.load(inp/'reference.npz');n=meta['frames'];frames.update((c['recording_group'],f) for f in range(c['source_start_frame'],c['source_end_frame_exclusive']))
  assert len(align)==len(z['R'])==n==c['source_end_frame_exclusive']-c['source_start_frame'];assert [a['source_frame'] for a in align]==list(range(c['source_start_frame'],c['source_end_frame_exclusive']))
  assert all(a['status']=='valid' and a['quality']>=.5 and a['nearest_gap_ms']<=30 for a in align);assert len({a['graph'] for a in align})==1
  assert np.all((np.diff(z['times_ns'])/1e9>=.08)&(np.diff(z['times_ns'])/1e9<=.12));assert all(np.isfinite(z[k]).all() for k in z.files);assert np.max(np.abs(z['R'].transpose(0,2,1)@z['R']-np.eye(3)))<1e-5
  for p,h in meta['hashes'].items():assert sha(inp/p)==h,(cid,p);hasheschecked+=1
  assert meta['pose_held_frames']==0;poses+=n;maxgap=max(maxgap,max(a['nearest_gap_ms'] for a in align))
  for page in read(OUT/'mask_review'/f'{cid}_index.json'):assert page['page'] in visual['reviewed_contact_pages'],page
  media=read(PRESENT/'clips'/f'{cid}_media.json');videos.append((Path(media['video']),n))
 assert len(frames)==3118
 oldframes={(o['recording'],f) for o in old for f in range(o['start_frame'],o['end_frame_exclusive'])};assert len(frames-oldframes)==1813
 timeline=read(PRESENT/'timeline.json');videos+=[(PRESENT/'ALL_118_CLIPS.mp4',3118),(PRESENT/'ALL_WITH_MASKS.mp4',sum(r['frames'] for r in timeline if r['with_masks_start_s'] is not None))]
 for g in sorted({r['group'] for r in timeline}):
  rs=[r for r in timeline if r['group']==g];videos.append((Path(rs[0]['group_video']),sum(r['frames'] for r in rs)))
 for p,n in videos:
  cap=cv2.VideoCapture(str(p));count=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(1080,1280,3);count+=1
  fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==n and abs(fps-10)<1e-6,(str(p),count,n)
 result.update(valid_pose_frames=poses,max_nearest_pose_gap_ms=maxgap,pose_held_frames=0,input_hashes_checked=hasheschecked,decoded_review_videos=len(videos),unique_frames=len(frames),new_source_frames=len(frames-oldframes),all_mask_pages_reviewed=True)
 write(OUT/'verification.json',result);return result

def reports(s,windows,clips):
 REPORT.mkdir(parents=True,exist_ok=True);timeline=read(PRESENT/'timeline.json');by={r['clip']:r for r in timeline}
 def link(p,label):return f'[{label}](<{Path(p).as_posix()}>)'
 lines=['# 剩余 G1 标注：检查入口','',f"本轮处理全部剩余 **225 个候选时间窗口，118 段，19 个录制**。初筛保留 **{s['review_masks']:,} 个 mask**，其中 **{s['strict_person_windows']} 个人物×窗口**通过连续 21 帧及近人质量筛选，仍待你检查。",'',link(PRESENT/'ALL_WITH_MASKS.mp4','优先播放：有初筛 mask 的合集'),'',link(PRESENT/'ALL_118_CLIPS.mp4','播放全部 118 段，含排除或空片段'),'',link(REPORT/'RESULTS.md','统计与筛选报告'),'','按本轮片段编号 + Nxx 人物编号指出问题；这些编号与上一批无对应关系。','']
 for g in sorted({r['group'] for r in timeline}):
  rs=[r for r in timeline if r['group']==g];lines.append('- '+link(rs[0]['group_video'],f"第 {g} 组：{rs[0]['clip']}–{rs[-1]['clip']}"))
 lines+=['','| 片段 | mask 数 | 完整/严格人物窗口 | 视频 | 说明 |','|---|---:|---:|---|---|']
 for c in clips:
  ws=[w for w in windows if w['clip']==c['clip']];lines.append(f"| {c['clip']} | {c['review_masks']} | {sum(len(w['complete21_ids']) for w in ws)} / {sum(len(w['strict_candidate_ids']) for w in ws)} | {link(c['video'],'播放')} | {c['scene_exclusion'] or '待检查'} |")
 (OUT/'presentation/START_HERE.md').write_text('\n'.join(lines)+'\n','utf8')
 report=f'''# 剩余 G1 全覆盖标注与筛选（2026-10-07）

{link(OUT/'presentation/START_HERE.md','检查入口')}

## 完成范围

现有 835 个相机条件合格窗口中，492 已在旧片段中完整处理，56 与用户拒绝时间段重叠，62 与既有场景排除段重叠。本轮固定处理其余 225 个：77 个完全未覆盖，148 个带部分既有上下文。覆盖 19 个录制，合并 118 段、3,118 帧（311.8 秒），其中 1,813 帧（181.3 秒）是真正新增画面，1,305 帧为旧上下文。没有替换空样本，也没有根据预测误差选样。

这里的“全部”限于现有相机条件清单的剩余范围；并非全部原始录像、全部真实 G1 场景或所有行人的穷尽标注。此前用户拒绝的 13 个不同片段的时间范围仍排除。

## 本轮规模

| 项目 | 数量 |
|---|---:|
| 原始 SAM mask | {s['raw_masks']} |
| 展示用初筛 mask | {s['review_masks']} |
| 原始 / 初筛片段内 ID | {s['raw_clip_local_ids']} / {s['review_clip_local_ids']} |
| 有初筛 mask 的片段 | {s['clips_with_masks']} |
| 场景或目标整体排除片段 | {s['scene_excluded_clips']} |
| 连续 21 帧人物窗口候选 | {s['complete21_person_windows']} |
| 严格筛选后人物窗口 | {s['strict_person_windows']} |
| 严格候选覆盖时间窗口 / 片段 / 录制 | {s['strict_time_windows']} / {s['strict_clips']} / {s['strict_recordings']} |
| 严格候选片段内 ID | {s['strict_clip_local_ids']} |
| 同片段同 ID 内不重叠窗口 | {s['within_identity_nonoverlap']} |
| 同录制时间不重叠窗口 | {s['recording_time_nonoverlap']} |

ID 是片段内编号，不是独立真实人数；窗口可重叠，多人共享相机运动，不重叠也不等于统计独立。不能把本轮候选数量直接加到旧训练样本数：部分上下文重叠，未来合并还需按录制和身份片段去重、保留测试隔离。

严格候选中，12 个来自全新未覆盖窗口，118 个来自带部分旧上下文的新完整窗口。来自录制 192702 与 150809 的候选分别为 34 和 32 个，共占 50.8%；新增数据仍有录制集中性。

从 250 个连续候选筛至 130 个，排除了 120 个。其中 91 个有边缘截断，33 个存在可见身体质量问题，5 个存在被撤回负点供体依赖；原因有重叠，不能相加。完整窗口数量与独立场景覆盖需同时报告。

## 方法与质量筛选

沿用近人 V6 / SAM2.1 Hiera Large：显示高度 12% 入选、8% 保留，无总人数上限，最多三人一批，双向传播，Aria 参考旋转辅助，以及前批可靠当前帧 mask 的负点。尺度是图像显示高度，不是实测距离。

运行环境与上一批一致：SAM2 可选 CUDA 后处理扩展不可用，沿用原适配器跳过填孔后处理；没有为本轮改动模型或运行环境。

118 段全部进行了场景查看；所有实际送入 SAM 的 proposal 已查看时间条带，断裂、明显串人、反射、海报、局部重复及不属于当前 G1 的片段均有显式处理记录。实际 mask 又逐 ID 查看首、中、末及最弱提示一致性样本，共 {s['mask_review_pages']} 页；{s['quarantined_ids']} 个 ID 因明显 mask 错误隔离。展示 revision 为 review_v2，原始 mask、review_v1、所有检查图均保留。

逐 mask 自动筛选为提示 IoU≥0.45、框外比例≤0.10、非 V6_UNCERTAIN、高度≥8%，严重 mask 相交双方挂起。严格候选另要求同一 census segment 连续 21 帧、全程无图像边缘截断、通过抽样身体完整性检查、相机对齐合格。凡所在 segment 有被撤回负点供体依赖，整段候选排除；这是保守筛选，没有重跑记忆消除影响。保留的展示 mask 中 {s['donor_dependency_person_frames']} 人物帧带供体依赖标记。

短轨迹、出入画面、被遮挡、边缘截断仍可出现在检查视频里，但不自动成为预测训练样本。筛选依据标注/场景质量，不依据哪个模型预测得好。

## 交付

- 每段原始 masks_rle.jsonl.gz、decisions.jsonl.gz、census.json 和 segments/。
- 每段 review_v2/ 为当前初筛标注；presentation/review_v2/ 为编号视频、分组合集及全集。
- window_review_ledger.json 保留全部 225 个窗口的状态，strict_person_window_candidates.json 给出筛选候选；continuous21_person_window_candidates.json 保留未通过的逐项原因。
- dataset/strict_candidates.npz 与 metadata.json 是单独候选包，含 21 帧中心、框、参考旋转、内参、时间戳和来源；没有并入活动训练集。
- verification.json 记录锁定清单、旧资产哈希、输入同步/几何及最终视频完整解码检查。

这是经助手抽样检查的离线 SAM 参考，仍待用户验收，不是独立逐帧人工真值。本轮未训练预测模型、未变更旧数据/基线、未上传 CVAT。新增标注本身不能证明模型间的微小指标差异已经可靠。
'''
 (REPORT/'RESULTS.md').write_text(report,'utf8')
 payload=html.escape(json.dumps(s,ensure_ascii=False,indent=2))
 cards=[]
 for c in clips:
  rel=f"review_v2/clips/{c['clip']}_masks.mp4";cards.append(f'<tr><td>{c["clip"]}</td><td>{c["review_masks"]}</td><td><a href="{rel}">播放</a></td><td>{html.escape(c["scene_exclusion"] or "待检查")}</td></tr>')
 (OUT/'presentation/START_HERE.html').write_text('<!doctype html><meta charset="utf-8"><title>剩余 G1 标注检查</title><style>body{font-family:system-ui;margin:32px;max-width:1100px;background:#f5f7fa;color:#182536}video{width:100%}td{padding:8px;border-bottom:1px solid #ccc}a{color:#1462af}</style><h1>剩余 G1 标注检查</h1><p>本轮片段编号独立；初筛标注待验收。严格候选已单独筛选，尚未合并训练。</p><video controls preload="metadata" src="review_v2/ALL_WITH_MASKS.mp4"></video><h2>逐段视频</h2><table><tr><th>片段</th><th>mask 数</th><th>视频</th><th>备注</th></tr>'+''.join(cards)+'</table><details><summary>完整统计</summary><pre>'+payload+'</pre></details>','utf8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
 s,w,c=aggregate();reports(s,w,c)
 if a.verify:print(json.dumps(verify(),ensure_ascii=False),flush=True)
 print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
