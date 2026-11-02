"""Explicit visual selection; new ranges relative to the near-person dataset."""
from screen import *
import csv,hashlib
CHOSEN=[
 ('151153',25,30,'G1_candidate','站台沿直线跟随两位白上衣人物，前方主要目标清晰','两人相似衣着，需检查身份；首次增加该录制','151153_white_pair'),
 ('194648',145,150,'G1_candidate','站厅右侧黑上衣灰裤，低密度同向行走','少量侧向经过者及柱子；首次增加该录制','194648_black_follow'),
 ('192702',191,201,'G1_candidate','街道人行道白背心黑裤连续跟随','与旧01/03同一人物事件，必须同组划分','192702_white_top_follow'),
 ('193709',74,87,'G1_candidate_boundary','桥下至路口，黑上衣浅裤与浅色上衣人物持续跟随','夜间、前方路口、部分转头；与旧09同一黑衣人物事件','193709_black_follow'),
 ('164557',65,71,'G1_candidate_boundary','到入口前的平地，蓝上衣成人、白衣成人和孩子','家庭同行；截至上楼前，孩子携带物单独核查','164557_family_follow'),
 ('163303',3,8,'G1_candidate','商场平直通道，前方白衣及格纹上衣行人','少量迎面通过，末段离场边界核查；首次增加该录制','163303_corridor_pair'),
 ('165112',60,65,'G1_candidate_boundary','商场通道低交错主要目标','广告/橱窗人物需排除，末尾人流增加；首次增加该录制','165112_corridor'),
 ('205416',15,20,'G1_candidate','站台沿直线跟随粉色包人物与白衣背包人物','少量上车者和栏杆；首次增加该录制','205416_platform_follow'),
 ('150809',35,40,'G1_static_control','候车站台深色衣服看手机者，目标周围交互少','静止/缓动对照，不能充当自由行走数据','150809_waiting_phone'),
 ('193311',38,42,'G1_candidate_boundary','道路转角后两名迎面人物接近并离场','时间短、夜间，检查每人连续有效时长','193311_approaching_pair'),
 ('192702',169,181,'G1_candidate','人行道若干间隔经过的迎面行人与前方同向目标','近身离场时不强连身份，低照度','192702_approach_169_181')]


def main():
    old=json.loads((ROOT/'outputs/g1_street_candidates_20261001/selected_clips.json').read_text(encoding='utf8'))
    rows=[]
    for number,(sid,start,end,group,observation,limitations,episode) in enumerate(CHOSEN,11):
        assert not any(c['id']==sid and max(start,c['start_s'])<min(end,c['end_s']) for c in old)
        rec=base.INV[sid];sig=base.signals(sid);maps=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
        frame_map=[dict(output_frame=i,source_frame=f,**{k:v for k,v in maps[f].items() if k!='source_frame'}) for i,f in enumerate(range(start*10,end*10))]
        dest=OUT/'frame_maps'/f'{number:02d}.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(frame_map,ensure_ascii=False,indent=2),encoding='utf8')
        rows.append(dict(candidate=number,id=sid,start_s=start,end_s=end,duration_s=end-start,status=group,
                         visual_observation=observation,remaining_factors=limitations,episode_group=episode,recording_group=rec['recording'],
                         source_video=rec['video'],source_timestamp_map=rec['timestamp_map'],source_mps=rec['mps_source'],
                         source_start_frame=start*10,source_end_frame_exclusive=end*10,frame_map=str(dest),metrics=base.metrics(sig,start,end)))
    path=OUT/'selected_clips.json';assert not path.exists();path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    preserved=[p for p in (ROOT/'outputs/sam2_near_people_20261002').glob('*/final/*') if p.is_file()]
    preserved+=[ROOT/'scripts/sam2_g1_v6_v8_20261001/core.py',ROOT/'scripts/sam2_near_people_20261002/run.py']
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved},indent=2),encoding='utf8')
    print('Selected',len(rows),'clips,',sum(r['duration_s'] for r in rows),'seconds,',len({r['id'] for r in rows}),'recordings')


if __name__=='__main__':main()
