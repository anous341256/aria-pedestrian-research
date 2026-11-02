"""Verify completed exports and masks, and prepare review sheets without modifying raw outputs."""
import csv
import gzip
import json
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from cvat_sdk.masks import decode_mask
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
BASE=ROOT/'outputs/sam2_fullvideos_20260927'
REPORT=ROOT/'reports/sam2_fullvideos_20260927'
IDS=['162538','164557','151153','193709','201129']

def spans(frames):
    out=[]
    for f in frames:
        if out and out[-1][1]+1==f:out[-1][1]=f
        else:out.append([f,f])
    return out

def verify_one(sid):
    d=BASE/sid;sam=d/'sam2_raw_v1'
    if (sam/'verification.json').exists():return read_json(sam/'verification.json')
    m=read_json(d/'manifest.json');s=read_json(sam/'summary.json');tracks=read_json(sam/'tracks.json')['tracks']
    n=m['frames'];w=m['width'];h=m['height'];assert s['processed_frames']==n
    times=list(csv.DictReader((d/'frame_timestamps.csv').open()))
    assert len(times)==n and [int(r['source_rgb_index']) for r in times]==list(range(n))
    assert all(int(a['device_timestamp_ns'])<int(b['device_timestamp_ns']) for a,b in zip(times,times[1:]))
    lookup={};intervals={};boxcount=0
    for o in tracks:
        rows=o['frames'];assert [r['frame'] for r in rows]==list(range(o['prompt_frame'],n))
        for r in rows:
            assert r['source_rgb_index']==r['frame'] and not r['human_identity_confirmed'] and not r['human_box_geometry_confirmed']
            lookup[(o['id'],r['frame'])]=r
        present=[r['frame'] for r in rows if r['bbox_xyxy'] is not None];boxcount+=len(present)
        intervals[str(o['id'])]={'nonempty':spans(present),'empty':spans([r['frame'] for r in rows if r['bbox_xyxy'] is None])}
    seen=set();decoded=0
    with gzip.open(sam/'masks_rle.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            rle=json.loads(line);key=(rle['id'],rle['frame']);assert key not in seen;seen.add(key);r=lookup[key]
            data=rle['rle'];runs=data[:-4];x1,y1,x2,y2=data[-4:];x2+=1;y2+=1
            assert all(isinstance(v,int) and v>=0 for v in runs)
            assert 0<=x1<x2<=w and 0<=y1<y2<=h and sum(runs)==(x2-x1)*(y2-y1)
            assert sum(runs[1::2])==r['mask_pixels']
            expected=[x1,y1,x2,y2] if r['mask_pixels'] else None
            assert expected==r['bbox_xyxy']
            if rle['frame']%50==0 or rle['frame']==next(o['prompt_frame'] for o in tracks if o['id']==rle['id']):
                mask=decode_mask(data,image_width=w,image_height=h);yy,xx=np.where(mask)
                actual=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
                assert len(xx)==r['mask_pixels'] and actual==r['bbox_xyxy'];decoded+=1
    assert seen==set(lookup)
    xml=ET.parse(sam/'boxes_unreviewed.xml')
    for t in xml.findall('track'):
        oid=int(t.attrib['id']);o=next(o for o in tracks if o['id']==oid)
        present={int(b.attrib['frame']):b for b in t.findall('box') if b.attrib['outside']=='0'}
        assert set(present)=={r['frame'] for r in o['frames'] if r['bbox_xyxy'] is not None}
        for f,b in present.items():assert [float(b.attrib[k]) for k in ('xtl','ytl','xbr','ybr')]==lookup[(oid,f)]['bbox_xyxy']
    overview=sam/'overview_5s';overview.mkdir(exist_ok=True);tiles=[];page=0;overviewframes=[];videos=[]
    for filename,expected_sha in [(d/'rgb.mp4',m['video_sha256']),(sam/'comparison.mp4',s['comparison_sha256'])]:
        assert digest(filename)==expected_sha
        cap=cv2.VideoCapture(str(filename));count=0
        while True:
            ok,img=cap.read()
            if not ok:break
            if filename.name=='comparison.mp4' and (count%50==0 or count==n-1):
                overviewframes.append(count);tiles.append(cv2.resize(img[:,960:],(480,360)))
                if len(tiles)==12 or count==n-1:
                    while len(tiles)<12:tiles.append(np.zeros_like(tiles[0]))
                    cv2.imwrite(str(overview/f'page_{page:02d}.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,12,3)]),[cv2.IMWRITE_JPEG_QUALITY,94]);page+=1;tiles=[]
            count+=1
        cap.release();assert count==n;videos.append({'file':filename.name,'fully_decoded_frames':count,'sha256':expected_sha})
    src=Path(m['source_vrs']);assert (src.stat().st_size,src.stat().st_mtime_ns)==(m['source_size'],m['source_mtime_ns'])
    write_json(sam/'mask_intervals.json',intervals)
    v={'status':'passed','recording':sid,'full_source_frames':n,'videos':videos,'object_frame_records':len(lookup),'nonempty_boxes':boxcount,'all_rle_validated_algebraically':True,'sampled_full_resolution_masks_decoded':decoded,'all_xml_nonempty_frames_and_coordinates_match':True,'source_size_mtime_unchanged':True,'all_user_confirmation_flags_false':True,'overview_frames':overviewframes,'scope':'Technical integrity only, no tracking-accuracy claim.'}
    write_json(sam/'verification.json',v);print('VERIFIED',sid,boxcount,flush=True);return v

def index(results):
    REPORT.mkdir(exist_ok=True,parents=True)
    names={'162538':'食品卖场与购物通道','164557':'白天街道→商场','151153':'站台→扶梯→站厅','193709':'傍晚过街与人行道','201129':'夜间站外→站厅'}
    lines=['# 五段完整视频的 SAM2 跟踪结果','',f'5段源录制的全部RGB帧，共{sum(r["full_source_frames"] for r in results):,}帧，约{sum(r["full_source_frames"] for r in results)/600:.2f}分钟；每段5个指定对象，共25个ID。当前版本为未经后续纠正的SAM2自动传播，助手提示不等于用户确认。','', '每段视频保留完整时长，但并非每个目标从头到尾都可见，也未标注画面中所有人。目标首次提示前没有倒推标注，目标离开后不把ID分配给新人。第164557段S3是坐在拉杆箱上的儿童，应单列为非步行运动。','', '| 录制/场景 | 完整时长 | 处理帧 | 非空自动框数（非准确率） | 查看 |','|---|---:|---:|---:|---|']
    for r in results:
        sid=r['recording'];d=f'../../outputs/sam2_fullvideos_20260927/{sid}';n=r['full_source_frames']
        lines.append(f'| {sid} {names[sid]} | {n/10:.1f}秒 | {n} | {r["nonempty_boxes"]} | [完整对照视频]({d}/sam2_raw_v1/comparison.mp4) · [指定的5人]({d}/prompt_gallery.jpg) · [CVAT框文件]({d}/sam2_raw_v1/boxes_unreviewed.xml) |')
    lines+=['','[助手抽样复核与问题位置](REVIEW.md)','', '## 输出与限制','', '每段目录保留无标注RGB、VRS时间映射、mask RLE、逐帧框、初始提示、mask消失/重现等机械提示及每秒1帧的复核图。机械提示不等于已证实错误；非空框里也可能包含错误对象或背景。外接框表示可见预测mask范围，不表示补全的全身框；XML中的occluded字段没有作为人工遮挡标签使用。','', '模型为已有SAM2.1 Large与官方CVAT流式适配器。RGB预处理使用Aria出厂标定；不读取MPS或眼动。每个目标只给一次初始框，无后续补点、重新初始化或检测器替换。保留了缺少可选连通域后处理扩展时的实际运行配置。','', '技术校验覆盖全部RLE编码长度/像素数/范围、全部框坐标和XML一致性、全部视频完整解码与源索引映射，并抽样解码全分辨率mask。原始VRS大小/修改时间未变，全部用户身份与几何确认标记仍为false。检查通过不代表标注准确率合格。','', '本轮未导入/覆盖旧CVAT作业；现有SAM2服务保持运行。没有新增训练或未来轨迹预测结果。','', '[执行范围](EXECUTION.md) · [批次运行摘要](../../outputs/sam2_fullvideos_20260927/batch_summary.json) · [技术校验](../../outputs/sam2_fullvideos_20260927/verification.json)','']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    results=[]
    for sid in IDS:
        while not (BASE/sid/'sam2_raw_v1/summary.json').exists():time.sleep(5)
        results.append(verify_one(sid))
    write_json(BASE/'verification.json',{'status':'passed','recordings':results,'total_frames':sum(r['full_source_frames'] for r in results),'total_nonempty_automatic_boxes':sum(r['nonempty_boxes'] for r in results)})
    index(results);print('PACKAGE COMPLETE',flush=True)

if __name__=='__main__':main()
