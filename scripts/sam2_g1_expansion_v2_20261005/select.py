"""Visually selected unused time ranges, independent of prediction performance."""
from screen import *
import csv,hashlib
CHOSEN=[
 ('164557',35,50,'G1_candidate_boundary','白天人行道跟随蓝上衣成人、白衣成人及孩子','共享家庭事件；过近下肢裁切、孩子骑乘箱体单列','164557_family_follow'),
 ('151153',15,25,'G1_candidate','站台下车后两位浅色上衣人物同向行走','前段车门附近遮挡；与11号白衣双人事件同组','151153_white_pair'),
 ('194648',150,162,'G1_candidate_boundary','站厅黑上衣灰裤人物沿平地持续行走','柱子和短时迎面经过，转头较多；与12号同一事件','194648_black_follow'),
 ('163303',8,16,'G1_candidate','商场平直通道的迎面行人与同向目标','近身离场截断；不延伸到后续拥挤交叉','163303_corridor_pair'),
 ('193709',62,74,'G1_candidate_boundary','夜间人行道持续跟随黑上衣浅裤人物','目标交互较少但转头较多，保留G1/G2边界；与09/14同事件','193709_black_follow'),
 ('192702',215,221,'G1_candidate','傍晚街道人行道同向三人及迎面经过者','止于明显仰头之前；同录制与相邻片段归组','192702_group_215'),
 ('155313',178,190,'G1_candidate','白天有顶步行区多批迎面人物分段经过','反光玻璃、广告、短时近身出画需核查','155313_arcade_178'),
 ('201129',10,19,'G1_candidate_boundary','夜间站外有顶通道跟随灰上衣及前方双人','柱遮挡与头部倾斜，止于大转头前；新增录制','201129_arcade_follow'),
 ('165112',22,29,'G1_candidate_boundary','商场玻璃橱窗旁跟随浅色和深色上衣人物','广告人像及橱窗倒影排除，转头较多；新增录制','165112_corridor_pair'),
 ('155313',220,227,'G1_candidate','有顶步行区同向绿裤人物与迎面行人','反光广告、阳光变化与近身截断','155313_arcade_220')]
def main():
 old=[]
 for name in ['g1_street_candidates_20261001','sam2_g1_expansion_20261005']:
  old+=json.loads((ROOT/'outputs'/name/'selected_clips.json').read_text('utf8'))
 rows=[]
 for number,(sid,start,end,group,observation,limitations,episode) in enumerate(CHOSEN,22):
  assert not any(c['id']==sid and max(start,c['start_s'])<min(end,c['end_s']) for c in old)
  rec=base.INV[sid];maps=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  frames=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
  dest=OUT/'frame_maps'/f'{number:02d}.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(frames,indent=2),encoding='utf8')
  rows.append(dict(candidate=number,id=sid,start_s=start,end_s=end,duration_s=end-start,status=group,visual_observation=observation,remaining_factors=limitations,episode_group=episode,recording_group=rec['recording'],source_video=rec['video'],source_timestamp_map=rec['timestamp_map'],source_mps=rec['mps_source'],source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=base.metrics(base.signals(sid),start,end)))
 p=OUT/'selected_clips.json';assert not p.exists();p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 protected=[]
 for version in ['sam2_near_people_20261002','sam2_g1_expansion_20261005']:
  protected += [p for p in (ROOT/'outputs'/version).glob('*/final/*') if p.is_file()]
 protected += list((ROOT/'scripts/sam2_near_people_20261002').glob('*.py')) + list((ROOT/'scripts/sam2_g1_expansion_20261005').glob('*.py'))
 protected += [p for p in (ROOT/'models/forecasting/biflow_comp_dual_baseline_20261002').rglob('*') if p.is_file()]
 protected += [p for p in (ROOT/'outputs/transformer_expanded_20261005/data').glob('*') if p.is_file()]
 (OUT/'source_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2),encoding='utf8')
 print('Selected',len(rows),'clips',sum(r['duration_s'] for r in rows),'seconds')
if __name__=='__main__':main()
