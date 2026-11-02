from select import *
EXTRA=[
 ('155313',79,85,'明亮站厅近处站立及通行人物','完整身体、稳定窗口；遮挡断开'),
 ('155313',97,104,'站厅柱旁人物及迎面行人','站立与行走分别标记，排除箱包和重叠遮挡'),
 ('155313',168,173,'有顶平地通道迎面人物','连续身体及平稳窗口'),
 ('165112',117,123,'商场跟随白上衣黑背包人物','后段被迎面人物遮挡前截断'),
 ('165112',148,153,'商场跟随白上衣黑背包人物','与同录制其他时段共用录制分组，不能视为独立人'),
 ('194648',199,203,'商场窗旁静止人物','坐姿遮挡与站立完整身体分别核查')]
def main():
 p=OUT/'selected_clips.json';rows=json.loads(p.read_text('utf8'))
 if any(r['candidate']==42 for r in rows):return
 (OUT/'selected_clips_initial10.json').write_bytes(p.read_bytes())
 used=json.loads((OUT/'excluded_intervals.json').read_text('utf8'))+rows
 for number,(sid,start,end,observation,limits) in enumerate(EXTRA,42):
  assert not any(c['id']==sid and max(start,c['start_s'])<min(end,c['end_s']) for c in used)
  rec=base.INV[sid];maps=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  frames=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
  dest=OUT/'frame_maps'/f'{number:02d}.json';dest.write_text(json.dumps(frames,indent=2),encoding='utf8')
  row=dict(candidate=number,id=sid,start_s=start,end_s=end,duration_s=end-start,status='candidate_pending_window_level_G1_review',visual_observation=observation,remaining_factors=limits,episode_group=f'{sid}_unused_{start}',recording_group=rec['recording'],source_video=rec['video'],source_timestamp_map=rec['timestamp_map'],source_mps=rec['mps_source'],source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=base.metrics(base.signals(sid),start,end))
  rows.append(row);used.append(row)
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');print('Candidates',len(rows),flush=True)
if __name__=='__main__':main()
