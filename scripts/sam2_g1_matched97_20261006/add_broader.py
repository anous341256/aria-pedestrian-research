from select import *
EXTRA=[('180512',38,45,'商场中庭平地通行人物','以步行人物为主，遮挡与近身截断窗口排除'),
       ('180512',17,22,'商场中庭迎面两名行人','只保留完整身体窗口，近身出画截断'),
       ('193956',22,26,'夜间路口同向步行人物','低照度和截断严格检查，不补不可见身体')]
def main():
 p=OUT/'selected_clips.json';rows=json.loads(p.read_text('utf8'))
 if any(r['candidate']==48 for r in rows):return
 (OUT/'selected_clips_supplement16.json').write_bytes(p.read_bytes());used=json.loads((OUT/'excluded_intervals.json').read_text('utf8'))+rows
 for number,(sid,start,end,observation,limits) in enumerate(EXTRA,48):
  assert not any(c['id']==sid and max(start,c['start_s'])<min(end,c['end_s']) for c in used)
  rec=base.INV[sid];maps=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  frames=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
  dest=OUT/'frame_maps'/f'{number:02d}.json';dest.write_text(json.dumps(frames,indent=2),encoding='utf8')
  row=dict(candidate=number,id=sid,start_s=start,end_s=end,duration_s=end-start,status='candidate_pending_window_level_G1_review',visual_observation=observation,remaining_factors=limits,episode_group=f'{sid}_unused_{start}',recording_group=rec['recording'],source_video=rec['video'],source_timestamp_map=rec['timestamp_map'],source_mps=rec['mps_source'],source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=base.metrics(base.signals(sid),start,end))
  rows.append(row);used.append(row)
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');print('Candidates',len(rows))
if __name__=='__main__':main()
