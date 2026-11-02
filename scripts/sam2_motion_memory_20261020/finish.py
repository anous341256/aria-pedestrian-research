"""Aggregate diagnostics, audit immutable inputs and package comparison videos."""
import json,collections,time,subprocess,sys
from pathlib import Path
import numpy as np
from selection import ROOT,OUT,read,write,digest
from render import load,concat,verify
ARMS=['visual','external_box','camera','camera_person']
def main():
 revision=OUT/'seed_review_revision.json'
 if revision.exists() and read(revision).get('active',False) and not (OUT/'seed_repairs_complete.json').exists():
  clip=read(revision)['clip'];archive=OUT/'before_nonperson_seed_review';archive.mkdir(exist_ok=True)
  original=OUT/clip;destination=archive/clip
  assert original.resolve().is_relative_to(OUT.resolve()) and destination.resolve().is_relative_to(OUT.resolve())
  if original.exists() and not destination.exists():original.rename(destination)
  for variant in ['raw','final']:
   old=OUT/'presentation'/variant/(clip+'.mp4')
   target=archive/'presentation'/variant/old.name;target.parent.mkdir(parents=True,exist_ok=True)
   if old.exists() and not target.exists():old.rename(target)
  subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--clips',clip],check=True)
  from render import render_clip
  render_clip(next(c for c in read(OUT/'selection.json') if c['id']==clip))
  import audit_inputs
  audit_inputs.main()
  write(OUT/'seed_repairs_complete.json',dict(clip=clip,all_four_arms_rerun=True,original_outputs_archived=True))
 import numeric_audit
 numeric_audit.main()
 selection=read(OUT/'selection.json');stats={a:[] for a in ARMS};changes=[]
 for c in selection:
  arms={}
  for a in ARMS:
   dest=OUT/c['id']/a;complete=read(dest/'complete.json')
   logs=load(dest/'decisions.jsonl.gz');masks=load(dest/'masks_rle.jsonl.gz')
   frames=[x for x in logs if 'raw_candidates' in x]
   raw=[next((r for r in x['raw_candidates'] if r['source']==x['raw_source']),x['raw_candidates'][0]) for x in frames]
   row=dict(clip=c['id'],recording=c['recording'],propagated=len(frames),masks=len(masks),
      repairs=sum(x['status'].startswith('OFFLINE_REPAIR') for x in frames),
      pending=sum(x['status']=='PENDING_REVIEW' for x in frames),
      uncertain=sum(x['status'].startswith('UNCERTAIN') for x in frames),
      raw_valid=sum(x['valid'] for x in raw),
      raw_guide_iou_sum=sum(x['guide_iou'] for x in raw),
      runtime_seconds=complete['seconds'])
   stats[a].append(row);arms[a]={ (x['segment'],x['frame']):x['rle'] for x in masks }
  for a in ARMS[1:]:
   keys=set(arms['visual'])|set(arms[a])
   changes.append(dict(clip=c['id'],arm=a,total=len(keys),identical=sum(arms['visual'].get(k)==arms[a].get(k) for k in keys)))
 totals={}
 for a,rows in stats.items():
  total={k:sum(r[k] for r in rows) for k in ['propagated','masks','repairs','pending','uncertain','raw_valid','raw_guide_iou_sum','runtime_seconds']}
  n=max(1,total['propagated'])
  total.update(repair_fraction=total['repairs']/n,pending_fraction=total['pending']/n,
      raw_valid_fraction=total['raw_valid']/n,mean_raw_proposal_box_iou=total['raw_guide_iou_sum']/n)
  byrec=collections.defaultdict(list)
  for r in rows:byrec[r['recording']].append(r)
  total['recording_equal_repair_fraction']=float(np.mean([sum(x['repairs'] for x in rr)/max(1,sum(x['propagated'] for x in rr)) for rr in byrec.values()]))
  totals[a]=total
 write(OUT/'diagnostics.json',dict(totals=totals,per_clip=stats,mask_changes=changes,
   caveat='Proposal agreement/acceptance/repair diagnostics, NOT mask accuracy, recall or verified identity switches. No independent dense GT.'))
 contrasts={}
 for ref,new in [('visual','external_box'),('visual','camera'),('visual','camera_person'),('camera','camera_person')]:
  per_recording=collections.defaultdict(lambda:dict(propagated=0,reference_repairs=0,new_repairs=0))
  per_clip=[]
  for x,y in zip(stats[ref],stats[new]):
   assert x['clip']==y['clip'] and x['propagated']==y['propagated']
   r=per_recording[x['recording']]
   r['propagated']+=x['propagated'];r['reference_repairs']+=x['repairs'];r['new_repairs']+=y['repairs']
   per_clip.append(dict(clip=x['clip'],difference=y['repairs']-x['repairs']))
  def signs(values):
   return dict(fewer=sum(v<0 for v in values),same=sum(v==0 for v in values),more=sum(v>0 for v in values))
  contrasts[new+'_minus_'+ref]=dict(
   clip_signs=signs([r['difference'] for r in per_clip]),
   recording_signs=signs([r['new_repairs']-r['reference_repairs'] for r in per_recording.values()]),
   per_recording=dict(per_recording),per_clip=per_clip)
 write(OUT/'paired_repair_diagnostics.json',dict(contrasts=contrasts,
    caveat='Paired repair-demand diagnostics, not verified identity accuracy. Clips are not statistically independent.'))
 checks={};old=read(OUT/'protected_inputs.json')
 changed=[p for p,h in old.items() if digest(ROOT/p)!=h]
 checks['protected_files']=len(old);checks['changed_protected_files']=changed
 checks['code_hashes_match']=all(digest(ROOT/p)==h for p,h in read(OUT/'frozen_code.json').items())
 checks['all_arms_complete']=all((OUT/c['id']/a/'complete.json').exists() for c in selection for a in ARMS)
 checks['visual_and_box_have_no_biased_attention_calls']=all(
   read(OUT/c['id']/a/'complete.json')['biased_attention_calls']==0
   for c in selection for a in ['visual','external_box'])
 vendor=read(OUT/'preflight.json')['vendor_source_hashes']
 checks['vendor_sources_unchanged']=all(digest(Path(p))==h for p,h in vendor.items())
 initial=OUT/'initial_execution_before_raw_export'
 replay={}
 for cid in ['001','002']:
  for a in ARMS:
   p=initial/cid/a/'masks_rle.jsonl.gz'
   if p.exists():replay[cid+'_'+a]=load(p)==load(OUT/cid/a/'masks_rle.jsonl.gz')
 checks['pilot_output_replay_after_export_only_change']=replay
 # Selection independent of outputs, all intervals disjoint and excluded times avoided.
 checks['overlap_pairs']=sum(x['recording']==y['recording'] and x['absolute_start']<y['absolute_end'] and y['absolute_start']<x['absolute_end']
   for i,x in enumerate(selection) for y in selection[:i])
 forbidden=read(OUT/'forbidden_intervals.json')
 checks['forbidden_overlaps']=sum(x['recording']==y['recording'] and x['absolute_start']<y['end'] and y['start']<x['absolute_end'] for x in selection for y in forbidden)
 checks['camera_screen_pass']=all(c['camera_stats']['max_1s_rotation_deg']<=20 and c['camera_stats']['omega_p95_deg_s']<=50 for c in selection)
 checks['independent_attention_audit']=read(OUT/'independent_attention_audit.json')
 checks['input_audit']=read(OUT/'input_audit.json')
 if (OUT/'seed_review_revision.json').exists():
  checks['seed_review_resolution']=read(OUT/'seed_review_revision.json')
  assert checks['seed_review_resolution']['same_as_initial_selection']
  assert digest(OUT/'selection.json')==checks['seed_review_resolution']['final_selection_sha256']
 checks['numeric_backend_audit']=read(OUT/'numeric_backend_audit.json')
 if (OUT/'parallel_retry.json').exists():
  checks['parallel_resource_retry']=read(OUT/'parallel_retry.json')
 import audit_exports
 checks['export_audit']=audit_exports.main()
 pres=OUT/'presentation';media=[]
 for variant in ['raw','final']:
  paths=[pres/variant/(c['id']+'.mp4') for c in selection]
  for p in paths:
   v=verify(p);assert v['frames']==21,(p,v);media.append(v)
  concat(paths,pres/f'ALL_{variant}.mp4')
  v=verify(pres/f'ALL_{variant}.mp4');assert v['frames']==21*len(selection);media.append(v)
  fixed=[c for c in selection if int(c['id'])<=21]
  concat([pres/variant/(c['id']+'.mp4') for c in fixed],pres/f'FIRST_21_RECORDINGS_{variant}.mp4')
  v=verify(pres/f'FIRST_21_RECORDINGS_{variant}.mp4');assert v['frames']==21*len(fixed);media.append(v)
  for start in range(0,len(selection),25):
   batch=selection[start:start+25]
   target=pres/f'group_{start//25+1:02d}_{variant}.mp4'
   concat([pres/variant/(c['id']+'.mp4') for c in batch],target)
   v=verify(target);assert v['frames']==21*len(batch);media.append(v)
 write(OUT/'media_verification.json',media);write(OUT/'verification.json',checks)
 assert not changed and checks['code_hashes_match'] and checks['vendor_sources_unchanged']
 assert checks['visual_and_box_have_no_biased_attention_calls']
 assert checks['overlap_pairs']==0 and checks['forbidden_overlaps']==0 and checks['camera_screen_pass']
 assert all(replay.values()),'Export-only pilot replay changed; inspect before delivery'
 import inspect_outputs
 inspect_outputs.main()
 import review_contrasts
 review_contrasts.main()
 report=ROOT/'reports/sam2_motion_memory_20261020/RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['SAM2 内部运动记忆 attention：第一版四组对照','',
  '范围：191 段、21 个录制、154 个旧素材片段来源；每段21帧，合计4011帧/401.1秒原速播放。',
  '所有选段在同一录制内不重叠。区间非统计独立，1017是目标段实例，不是独立真实人数。',
  '1017个目标段中386个覆盖完整21帧，631个短于21帧（其中19个仅1帧）；短暂进出保留，未补成完整轨迹。',
  '只标注近人，12%画面高度纳入、8%保留，无总人数上限；沿用已有身份提案与离线修复。',
  'SAM2.1 Hiera Large 权重不变，未训练。相机旋转来自Aria参考，未做相机平移几何补偿。',
  'A视觉记忆；B外部运动框；C内部旋转；D内部旋转+目标残余运动。四组使用共同视觉筛选规则。',
  'B是公平配对的外部框对照，不是完整旧V6复现。原V6运动评分被共同去除。',
  '复查15个有历史隔离标记的起始目标。050/N05单帧曾被误判为设施，查看连续画面后撤回；原始选样清单已逐字节恢复，未移除目标或修改mask。此例初始化身份仍需人工核查。',
  '内部attention偏置强度1.5预先固定，未按结果调参。前批负点来自各组自己的输出，因此比较包含闭环传播影响。',
  '各组运行耗时包含串行和并行调度，且不含共享图像编码，不能直接作为严格速度基准。',
  '一并行进程在083的内部attention计算中显存不足；完整保留已完成arm，后续用原始串行runner原样重试。未缩图、删人或改变参数。',
  'complete.json中的max_bias是该工作进程累计峰值；是否实际启用偏置以该组biased_attention_calls为准。',
  '', '以下为自动诊断，不是人工GT准确率：',
  '组别 | 非初始化传播次数 | 原始候选通过数 | 离线修复数 | 待复核数 | 输出mask数 | 传播耗时秒']
 for a,t in totals.items():
  lines.append(f"{a} | {t['propagated']} | {t['raw_valid']} | {t['repairs']} | {t['pending']} | {t['masks']} | {t['runtime_seconds']:.1f}")
 flags=checks['export_audit']['seed_quality_flags']
 lines+=['',f"共享起始mask自动质量标记：{len(flags)}个。保留为待复核，不据结果删样或改变四组输入。"]
 for flag in flags:
  lines.append(f"  {flag['clip']}/{flag['segment']}: SAM score={flag['sam_score']:.4f}, pixels={flag['quality']['pixels']}, below_retention={flag['below_retention']}")
 lines+=['数值后端检查（原生attention与全零dense bias，区分计算后端舍入与运动引导）：']
 for name,check in checks['numeric_backend_audit'].items():
  if isinstance(check,dict):
   lines.append(f"  {name}: zero-bias mask pixels changed={check['zero_dense_changed_mask_pixels']}; motion-vs-zero-bias changed={check['motion_vs_zero_dense_changed_mask_pixels']}")
 lines+=['','不能仅凭修复减少、提案框一致性增加或与旧mask更相似，宣称身份切换减少或精度提高。',
  '查看raw视频检查纠正前的传播；final视频检查共同修复后的实际标注。R表示该帧触发了修复。',
  'raw是当前一步修复前的候选，仍受之前帧已执行修复后的记忆影响；不是另一次完全关闭修复的连续运行。',
  '每段仅2.0秒首尾跨度，不能代表长时间遮挡恢复能力。不同段切开重置记忆，未建立跨段全局身份。',
  '已有提案范围约束可能掩盖长时间漂移，原始候选和修复次数已保留。',
  '相机仅补偿旋转，目标残余运动仍含平移/视差/噪声。MPS和SAM输入为离线数据，不宣称在线性能。',
  '',f"输入保护核验：{len(old)}项，修改0项；原始vendor代码未改；全部视频逐帧解码通过。",
  '第一批执行仅补充raw输出导出和分组计数后重新执行，预测mask回放一致。',
  '没有合并预测训练集、替换旧标签、上传CVAT或修改PPT。',
  '', '观看入口：outputs/sam2_motion_memory_20261020/presentation/START_HERE.txt']
 report.write_text('\n'.join(lines),encoding='utf8')
 (pres/'START_HERE.txt').write_text(
  '先看 FIRST_21_RECORDINGS_raw.mp4：21个录制各首段，约88.2秒，显示自动纠正前的差异。\n'
  '再看 FIRST_21_RECORDINGS_final.mp4：共同纠正后的标注。\n'
  'ALL_raw.mp4 / ALL_final.mp4：191段完整合集，各802.2秒，0.5倍速。\n'
  'group_01..08_raw/final.mp4：每组最多25段；raw/和final/下按001..191提供单段。\n'
  '四格顺序：左上原生视觉记忆；右上外部运动框；左下内部旋转；右下内部旋转+目标运动。\n'
  '相同颜色对应同一个片段内提案ID。R=触发自动修复，?=不确定；不是人工GT。\n'
  'raw显示当前一步修复前的候选；此前帧修复仍影响后续记忆，不是完全关闭修复的一组实验。\n'
  '不要把提案一致性或修复数当作真实精度。保留所有191段，包括空结果与失败。\n',encoding='utf8')
 write(OUT/'delivery_complete.json',dict(clips=len(selection),videos=len(media),report=str(report.relative_to(ROOT)),verification_passed=True))
 import summarize
 summarize.main()
 print('DELIVERY COMPLETE',json.dumps(totals),flush=True)
if __name__=='__main__':main()
