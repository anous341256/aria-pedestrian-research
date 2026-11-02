from screen import *
import csv,hashlib
CHOSEN=[
 ('164557',94,102,'商场餐饮通道跟随浅上衣及黑衣人物','只保留转头前后各自平稳的完整窗口；后段遮挡另排除'),
 ('151153',31,45,'站台同向行走人物','近身截断和后段人群需要逐目标筛选；不含扶梯'),
 ('194648',173,181,'站厅提包同向人物及迎面人物','连续完整身体和局部稳定朝向窗口才进入G1'),
 ('163303',17,32,'平地商场通道迎面及同向人物','短目标保留标注但不能组成完整窗口则不用'),
 ('193709',32,40,'夜间平地路口同向及迎面人物','弱光、远近变化；排除目标交叉遮挡窗口'),
 ('192702',146,154,'傍晚街道迎面白上衣成人与儿童','优先成人连续身体；儿童遮挡不强补'),
 ('155313',198,206,'有顶步行通道同向及迎面人物','只选择目标自身连续且交互少的窗口'),
 ('201129',49,55,'站厅入口连续可见人物','此片整体仍有转头；仅窗口级稳定且无重叠遮挡者可进入G1'),
 ('165112',50,58,'商场通道跟随浅上衣人物','橱窗人像和倒影排除；转头窗口过滤'),
 ('155313',156,164,'通往出口的平地通道同向人物','出口附近交叉前截断，近身出画不强补')]
def main():
 used=json.loads((OUT/'excluded_intervals.json').read_text('utf8'));rows=[]
 for number,(sid,start,end,observation,limits) in enumerate(CHOSEN,32):
  assert not any(c['id']==sid and max(start,c['start_s'])<min(end,c['end_s']) for c in used)
  rec=base.INV[sid];maps=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  frames=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
  dest=OUT/'frame_maps'/f'{number:02d}.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(frames,indent=2),encoding='utf8')
  rows.append(dict(candidate=number,id=sid,start_s=start,end_s=end,duration_s=end-start,status='candidate_pending_window_level_G1_review',visual_observation=observation,remaining_factors=limits,episode_group=f'{sid}_unused_{start}',recording_group=rec['recording'],source_video=rec['video'],source_timestamp_map=rec['timestamp_map'],source_mps=rec['mps_source'],source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=base.metrics(base.signals(sid),start,end)))
 p=OUT/'selected_clips.json';assert not p.exists();p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 protected=[]
 for version in ['sam2_near_people_20261002','sam2_g1_expansion_20261005','sam2_g1_expansion_v2_20261005']:
  protected += [p for p in (ROOT/'outputs'/version).glob('*/final/*') if p.is_file()]
  protected += list((ROOT/'scripts'/version).glob('*.py'))
 protected += [p for p in (ROOT/'models/forecasting/biflow_comp_dual_baseline_20261002').rglob('*') if p.is_file()]
 for version in ['transformer_expanded_20261005','transformer_expanded_v2_20261005']:
  protected += [p for p in (ROOT/'outputs'/version/'data').glob('*') if p.is_file()]
 (OUT/'source_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2),encoding='utf8')
 print('Selected',len(rows),'candidate intervals',sum(r['duration_s'] for r in rows),'seconds')
if __name__=='__main__':main()
