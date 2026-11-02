"""Create scene catalog and mechanical validation without changing inference output."""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cvat_sdk.masks import decode_mask

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
OUT=ROOT/'outputs/sam2_scene_trial_20260927'
REPORT=ROOT/'reports/sam2_scene_trial_20260927'
CLIP=OUT/'162538_020_040'
SAM=CLIP/'sam2_raw_v1'

def intervals(values):
    spans=[]
    for f in values:
        if spans and f==spans[-1][1]+1:spans[-1][1]=f
        else:spans.append([f,f])
    return spans

def build_catalog():
    REPORT.mkdir(parents=True,exist_ok=True)
    notes=read_json(Path(__file__).with_name('scene_notes.json'))
    inventory=read_json(OUT/'current_vrs_inventory.json')
    old=list(csv.DictReader((ROOT/'reports/Aria_pedestrian_screening/recording_screen.csv').open(encoding='utf-8-sig')))
    previous={r['recording']:r for r in old}
    raw={r['recording']:r for r in inventory['recordings']}
    rows=[]
    for note in notes:
        name='unknown_20260803_'+note['id'];r=raw[name]
        rgb=next(s for s in r['streams'] if s['label']=='camera-rgb')
        p=previous[name]
        sheet=ROOT/'reports/Aria_pedestrian_screening/contact_sheets'/f'{name}.jpg'
        assert sheet.is_file()
        rows.append(dict(note,recording=name,source_vrs=str(ROOT/'data/Aria/raw_vrs'/f'{name}.vrs'),vrs_rgb_records=rgb['data_records'],rgb_timestamp_span_s=(rgb['last_device_time_ns']-rgb['first_device_time_ns'])/1e9,prior_export_frames=int(p['frames']),prior_export_duration_s=float(p['duration_sec']),visual_review_basis='assistant reviewed all 12 previously sampled frames per recording, not full video',contact_sheet=str(sheet),historical_split_not_adopted=p['split']))
    write_json(OUT/'scene_catalog.json',rows)
    with (OUT/'scene_catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=Counter(r['group'] for r in rows)
    totals={'recordings':len(rows),'vrs_rgb_records':sum(r['vrs_rgb_records'] for r in rows),'vrs_timestamp_span_sum_s':sum(r['rgb_timestamp_span_s'] for r in rows),'previous_export_frames':sum(r['prior_export_frames'] for r in rows),'previous_export_duration_s':sum(r['prior_export_duration_s'] for r in rows),'visual_sample_frames':len(rows)*12,'primary_group_counts':dict(groups),'note':'Groups describe dominant routes, not mutually exclusive frame-level scene-duration statistics.'}
    write_json(OUT/'scene_summary.json',totals)
    lines=['# 现有Aria视频场景盘点（2026-09-27）','',f"共 **{len(rows)} 段录制，约 {totals['previous_export_duration_s']/60:.1f} 分钟**。本次读取29个当前VRS的元数据，并逐张查看已有的29张抽样图（每段12帧，共348帧）。场景标签为助手视觉判断，**没有逐帧看完全部47.6分钟**；短暂事件、精确场景边界和具体让行行为仍需连续复核。",'',f"当前VRS共有 **{totals['vrs_rgb_records']:,} 条RGB记录**；旧MP4导出为 **{totals['previous_export_frames']:,} 帧**。两者相差 {totals['vrs_rgb_records']-totals['previous_export_frames']}，差异的具体原因尚未逐帧追溯，不能把旧MP4帧号直接当当前VRS索引。新SAM2试跑直接读当前VRS并另存时间映射。",'', '## 总体构成','', '| 按主要路线归类 | 录制数 |', '|---|---:|']
    lines += [f'| {group} | {count} |' for group,count in groups.items()]
    lines += ['', '该归类用于检索，很多录制跨越多个场景；数量不代表各场景时长或独立统计样本数。所有文件名标记同一天、同一设备的连续活动，29段不等于29个独立环境。旧初筛CSV中的train/val/test仅是历史记录，本次不恢复该划分，也不改动中期实验划分。','', '## 逐段场景与研究价值','', '文件名前缀均为 `unknown_20260803_`。时长为已有RGB导出的约数，点击编号查看本次已核对的12帧抽样图。图中的蓝框是旧自动检测，不是人工标注；其中可能含广告、模特、反射或手机屏幕误检。','', '| 编号 | 秒 | 画面场景/路线 | 可见内容 | 主要难点 | 用途判断 |','|---|---:|---|---|---|---|']
    for r in rows:
        sheet=f'../Aria_pedestrian_screening/contact_sheets/{r["recording"]}.jpg'
        lines.append(f'| [{r["id"]}]({sheet}) | {r["prior_export_duration_s"]:.1f} | {r["scene"]} | {r["content"]} | {r["difficulty"]} | {r["use"]} |')
    lines += ['', '## 对当前素材的评价','', '- **优势：真实步行和相机变化足够丰富。** 不只有候车：商场跟随/擦肩、站厅双向客流、上下楼、闸机停走、夜间过街都存在。适合研究图像运动混合、遮挡、视角变化和有限范围的身份保持。','- **分布明显集中。** 车站、商场及附近道路占主导，大量片段来自相近路线。不能据此宣称覆盖一般手持/胸戴、车载或跨天气/跨城市泛化；抽样中也没有建立这些域的系统覆盖。','- **长录制不等于长可靠轨迹。** 多段低头看手机、手/头发入镜、近处背包挡住视野、转头出画；需要按目标可见范围选窗。','- **行人与相机都在运动的片段应单列。** 站台候车适合补偿机制，但不足以证明学会人的运动；162538、155313、193709及201129等包含更多真实步行。','- **低照度是可用的困难条件。** 192702、193311、193524、193709、193956呈现明显暗区与亮招牌反差；低照度会影响检测、mask、光流，不能把所有失败归因于身份关联。','- **车辆出现与真实交互应分开。** 白天/傍晚道路片段可见车辆，193311有停车场出入口环境，但是否存在明确驶出、让行或冲突事件，仍需连续画面确认。','', '## 本次新SAM2试跑','', '选择 **162538，当前VRS RGB索引200–399（约20–40秒）**，食品卖场通道。片段在查看SAM2输出前固定。三个对象由助手给框提示，第三个在局部第10帧加入；不使用YOLO、MPS或眼动。它是一次有目标提示的部分分割跟踪尝试，不是全场自动标注或未来轨迹预测。','', '[SAM2试跑结果](SAM2_TRIAL.md) · [场景数据表](../../outputs/sam2_scene_trial_20260927/scene_catalog.csv) · [原始VRS元数据清单](../../outputs/sam2_scene_trial_20260927/current_vrs_inventory.json)','']
    (REPORT/'SCENE_CATALOG.md').write_text('\n'.join(lines),encoding='utf-8')
    return totals

def scene_atlas():
    selections=[('150809',401,'站台候车'),('162538',472,'商场食品卖场'),('163506',79,'楼梯与扶梯'),('164557',443,'白天临街步行'),('193709',443,'傍晚路口过街'),('194427',61,'拥挤车厢')]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',26)
    atlas=Image.new('RGB',(1280,1548),(16,20,28))
    for i,(sid,f,title) in enumerate(selections):
        path=Path('/mnt/e/Aria/parsed')/f'unknown_20260803_{sid}'/'logs/214-1/video.mp4'
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read();cap.release();assert ok
        tile=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(640,480)),cv2.COLOR_BGR2RGB))
        x=(i%2)*640;y=(i//2)*516
        atlas.paste(tile,(x,y+36));ImageDraw.Draw(atlas).text((x+12,y+2),f'{title}  |  {sid}',font=font,fill='white')
    atlas.save(OUT/'scene_examples.jpg',quality=94)

def detailed_review():
    # Independent raw/overlay panels at diagnosed re-entry and overlap boundaries.
    cap=cv2.VideoCapture(str(CLIP/'rgb.mp4'))
    for group,frames in [('s1_dropout',[34,35,37,38,40,42,43,44]),('occlusion_identity',[57,58,66,67,94,95,105,121,131,180])]:
        tiles=[]
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,raw=cap.read();assert ok
            overlay=cv2.imread(str(SAM/'frames'/f'{f:03d}.jpg'))
            raw=cv2.resize(raw,(640,480));over=cv2.resize(overlay,(640,480))
            cv2.rectangle(raw,(0,0),(640,32),(0,0,0),-1)
            cv2.putText(raw,f'Original RGB | f={f} | source {20+f/10:.1f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            tiles.append(np.hstack([raw,over]))
        for p in range(0,len(tiles),4):
            cv2.imwrite(str(SAM/f'{group}_{p//4}.jpg'),np.vstack(tiles[p:p+4]))
    cap.release()

def verify():
    tracks=read_json(SAM/'tracks.json')['tracks'];manifest=read_json(CLIP/'manifest.json')
    lookup={(o['id'],r['frame']):r for o in tracks for r in o['frames']}
    count=0
    with gzip.open(SAM/'masks_rle.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line);r=lookup[(obj['id'],obj['frame'])]
            mask=decode_mask(obj['rle'],image_width=2560,image_height=1920)
            yy,xx=np.where(mask)
            b=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
            assert b==r['bbox_xyxy'] and len(xx)==r['mask_pixels']
            count+=1
    assert count==590
    video_results=[]
    for path in [CLIP/'rgb.mp4',SAM/'comparison.mp4']:
        cap=cv2.VideoCapture(str(path));frames=0
        while True:
            ok,img=cap.read()
            if not ok:break
            assert img is not None;frames+=1
        cap.release();assert frames==200
        video_results.append({'path':str(path),'fully_decoded_frames':frames,'sha256':digest(path)})
    ranges={str(o['id']):{'nonempty_intervals':intervals([r['frame'] for r in o['frames'] if r['bbox_xyxy']]),'empty_intervals':intervals([r['frame'] for r in o['frames'] if r['bbox_xyxy'] is None])} for o in tracks}
    write_json(SAM/'mask_intervals.json',ranges)
    tree=ET.parse(SAM/'boxes_unreviewed.xml')
    for o,t in zip(tracks,tree.findall('track')):
        present=[int(b.attrib['frame']) for b in t.findall('box') if b.attrib['outside']=='0']
        assert present==[r['frame'] for r in o['frames'] if r['bbox_xyxy'] is not None]
    vrs=Path(manifest['source_vrs'])
    assert vrs.stat().st_size==manifest['source_size'] and vrs.stat().st_mtime_ns==manifest['source_mtime_ns']
    assert len(list((SAM/'frames').glob('*.jpg')))==200
    assert all(not r['human_identity_confirmed'] and not r['human_box_geometry_confirmed'] for r in lookup.values())
    result={'status':'passed','mask_records_verified':count,'nonempty_mask_boxes':sum(r['bbox_xyxy'] is not None for r in lookup.values()),'videos':video_results,'review_frames':200,'review_sheets':13,'source_vrs_size_mtime_unchanged':True,'all_human_confirmation_flags_false':True,'scope':'Technical integrity only; not segmentation or identity accuracy.'}
    write_json(OUT/'verification.json',result)
    print(json.dumps({'verification':result,'intervals':ranges},ensure_ascii=False),flush=True)

if __name__=='__main__':
    print(json.dumps(build_catalog(),ensure_ascii=False),flush=True)
    scene_atlas();detailed_review();verify()
