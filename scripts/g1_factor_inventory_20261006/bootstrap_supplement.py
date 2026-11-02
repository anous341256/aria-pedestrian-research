from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2]
OLD='sam2_g1_matched97_20261006';NEW='sam2_g1_factor_supplement_20261006'
OUT=ROOT/'outputs'/NEW;HERE=ROOT/'scripts'/NEW
OUT.mkdir(exist_ok=True);HERE.mkdir(exist_ok=True)
chosen=[1,3,6,10,12,13,21,28,33,39,48,53,56,63]
rows=json.loads((ROOT/'outputs/g1_factor_inventory_20261006/unused_screen_candidates.json').read_text('utf8'))
inv={r['id']:r for r in json.loads((ROOT/'outputs/dataset_threeway_20261001/motion_inventory.json').read_text('utf8'))}
selected=[]
for cid,sid in enumerate(chosen,55):
 r=next(x for x in rows if x['screen_id']==sid);v=inv[r['id']]
 maps=list(csv.DictReader(open(v['timestamp_map'],encoding='utf-8-sig')))
 start,end=r['start_s'],r['end_s']
 frames=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
 dest=OUT/'frame_maps'/f'{cid:02d}.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(frames,indent=2),encoding='utf8')
 selected.append(dict(candidate=cid,id=r['id'],start_s=start,end_s=end,duration_s=end-start,status='candidate_pending_window_level_G1_review',screen_id=sid,visual_observation='Level-ground station, passage, mall or street; near continuous people reviewed individually after rectification.',remaining_factors='Only complete clear target windows meeting G1 camera screen are admitted; no stairs/escalators; this is not an exhaustive census.',episode_group=f"{r['id']}_unused_{start}",recording_group=v['recording'],source_video=v['video'],source_timestamp_map=v['timestamp_map'],source_mps=v['mps_source'],source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=r))
p=OUT/'selected_clips.json';assert not p.exists();p.write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')
for name in ['prepare_input.py','discover.py','inventory.py','census.py','temporal.py','prepare_census.py','run.py','qa.py','export.py','final_review.py','finish_labels.py','render_final.py','clip_list.py','window_motion.py']:
 p=HERE/name;assert not p.exists()
 s=(ROOT/'scripts'/OLD/name).read_text('utf8').replace(OLD,NEW).replace('range(32,42)','range(55,69)')
 s=s.replace('10 new intervals, 91 seconds','14 new intervals, 112 seconds').replace('annotated_seconds=91','annotated_seconds=112').replace('91 seconds','112 seconds').replace('All 910 frames','All 1120 frames').replace('91s','112s')
 p.write_text(s,encoding='utf8')
print('Prepared',len(selected),'candidate clips; original sources untouched.')
