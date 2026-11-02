"""Stage full videos and native CVAT track XML, preserving fixed SAM2 ID attributes."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/sam2_fullvideos_20260927'
DEST=BASE/'cvat_upload'
NAMES={'162538':'food_hall','164557':'day_street_mall','151153':'platform_escalator_hall','193709':'dusk_crosswalk','201129':'night_station'}
LABELS=[{'name':'person','color':'#33cc99','type':'rectangle','attributes':[{'name':'sam2_id','input_type':'text','mutable':False,'values':['unknown']}]}]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    DEST.mkdir(exist_ok=True)
    rows=[]
    for sid,name in NAMES.items():
        src=BASE/sid
        summary=json.loads((src/'sam2_raw_v1/summary.json').read_text(encoding='utf-8'))
        video=DEST/f'Aria_{sid}_{name}.mp4'
        if not video.exists():os.link(src/'rgb.mp4',video)
        assert os.path.samefile(src/'rgb.mp4',video)
        original=src/'sam2_raw_v1/boxes_unreviewed.xml';tree=ET.parse(original)
        label=tree.find('meta/task/labels/label');attrs=label.find('attributes')
        if attrs is None:attrs=ET.SubElement(label,'attributes')
        a=ET.SubElement(attrs,'attribute')
        for k,v in [('name','sam2_id'),('mutable','False'),('input_type','text'),('default_value','unknown'),('values','unknown')]:ET.SubElement(a,k).text=v
        count=0
        for track in tree.findall('track'):
            oid=int(track.attrib['id'])
            for box in track.findall('box'):
                ET.SubElement(box,'attribute',{'name':'sam2_id'}).text=f'S{oid}'
                count+=box.attrib['outside']=='0'
        assert count==sum(summary['nonempty_mask_frames'].values())
        xml=DEST/f'Aria_{sid}_SAM2_raw_boxes.xml';tree.write(xml,encoding='utf-8',xml_declaration=True)
        rows.append({'recording':sid,'video':str(video),'xml':str(xml),'frames':summary['processed_frames'],'tracks':5,'nonempty_boxes':count,'original_xml_sha256':digest(original),'upload_xml_sha256':digest(xml),'status':'staged','video_staging':'hard link to unchanged full-resolution RGB'})
    (DEST/'labels.json').write_text(json.dumps(LABELS,indent=2),encoding='utf-8')
    (DEST/'staging_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps([{'recording':r['recording'],'frames':r['frames'],'boxes':r['nonempty_boxes']} for r in rows]))

if __name__=='__main__':main()
