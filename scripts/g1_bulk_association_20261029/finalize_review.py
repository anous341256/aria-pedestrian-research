from bulk_common import *

notes=read(OUT/'visual_review_notes.json');notes['status']='completed_sampled_review';notes['reviewed_temporal_overviews']=sorted(read(OUT/'preview_ready/index.json')['clips']+read(OUT/'preview_remaining/index.json')['clips']);assert len(set(notes['reviewed_temporal_overviews']))==191
notes['reviewed_target_sample']=read(OUT/'review_target_sample/index.json')
notes['reviewed_mixed_source_windows']=read(OUT/'mixed_window_review/index.json')
notes['reviewed_final_comparison_midframes']=['056','105','191']
notes['final_review_summary']='逐段查看全部191段的首/中/末总览（573个场景时刻），另查每个有连续窗口的录制的首个目标，共20条首/中/末mask条带；检查全部6条跨来源ID连续窗口的变化前后图，并核对056/105/191最终对比图。此为抽样图像核查，不是逐帧播放或全量人工GT。022/120的背景误标疑点经原图放大后撤回，均为边缘局部人物。'
notes['notes'] += [
 dict(clip='104',track='T1',assessment='Raw temporal crops at frames0/3/5 show a real seated/crouched person beside a sign, not an isolated hand/background. Short track; no full forecasting window.'),
 dict(clip='141',track='T12',assessment='Full-resolution frame20 confirms the pink-shirt person is distinct from black-shirt foreground person. This does not establish identity continuity across all frames.'),
 dict(clip='005',track='T1',assessment='First-per-recording target sample passes automatic quality, but body extent changes as the close person reaches the image boundary. Stable ID is not proof of a stable physical body-center proxy.'),
 dict(clip='019',track='T3',assessment='First-per-recording sample retains person at ticket gate; lower-body visibility changes with gate occlusion. Automatic quality is not dense mask GT.')]
notes['mixed_window_assessment']={'011_T2_000':'Sampled transition appears to retain the same bag-carrying person; source ID changes do not by themselves establish a real switch.', '056_T7_000':'Sampled frames appear to show the same light-clothed woman across source N20->N19; automatic quality pass retained.', '101_T4_000':'Sampled N04->N06->N04 frames appear to follow the same patterned-shirt person; existing ambiguity flag retained.', '105_T2_000':'Sampled N11->N25 frames appear to follow the same white-shirt person after foreground occlusion; automatic quality pass retained.', '105_T10_000':'Sampled N13->N24 frames appear to follow the same blue-shirt person; existing ambiguity flag retained.', '122_T6_000':'Occluded target appears behind crossing foreground people; source masks fragmented and uncertain. Existing failed-quality status retained; not declared human-confirmed ID error.'}
notes['notes']=list({json.dumps(n,sort_keys=True):n for n in notes['notes']}.values());write(OUT/'visual_review_notes.json',notes)
a=read(OUT/'comparison.json');counts={}
for scope in ['per_clip','per_recording']:
 old=a['baseline'][scope];new=a['partial'][scope];metric='changes' if scope=='per_recording' else 'source_id_assignment_changes';counts[scope]=dict(improve=sum(new[k][metric]<old[k][metric] for k in old),tie=sum(new[k][metric]==old[k][metric] for k in old),worse=sum(new[k][metric]>old[k][metric] for k in old))
write(OUT/'comparison_direction.json',counts)
report=ROOT/'reports'/OUT.name/'RESULTS.txt'
with report.open('a',encoding='utf-8') as fp:
 fp.write('\n补充诊断：来源ID对应变化按片段与录制改善/持平/变差数量：'+json.dumps(counts,ensure_ascii=False)+'\n')
 fp.write('全部329窗口中6条跨旧来源ID，其中2条通过自动质量规则。已检查6条变化前后图：多数看起来是同一人的来源ID被拆分，不能把这些警报直接计为真实串身份。122存在遮挡与mask碎片疑点，原质量不通过状态保留。详见mixed_window_review与visual_review_notes.json。\n')
pack=read(OUT/'packaging_complete.json');pack['awaiting_final_visual_review']=False;write(OUT/'packaging_complete.json',pack)
print(counts)
