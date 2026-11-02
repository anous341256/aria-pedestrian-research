"""Versioned MOT/CVAT export with explicit missing-frame markers and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tracking'))
from audit_cvat_video_annotations import expand_track

ATTRIBUTES=['detector_score','association_status','node_key','track_anchor']


def export(directory,engine,result,labels,revision,reviews=None):
    dest=Path(directory)/'exports'/f'rev_{revision:06d}'
    if dest.exists():
        manifest=dest/'manifest.json'
        if not manifest.exists():raise RuntimeError('Incomplete prior export, use a new revision or inspect files')
        return str(dest)
    dest.mkdir(parents=True)
    meta=engine.input;root=ET.Element('annotations');ET.SubElement(root,'version').text='1.1'
    task=ET.SubElement(ET.SubElement(root,'meta'),'task')
    for k,v in [('id',0),('name','Aria identity corrected preannotation'),('size',meta['frames']),('mode','interpolation'),('overlap',0),('start_frame',0),('stop_frame',meta['frames']-1),('frame_filter','')]:ET.SubElement(task,k).text=str(v)
    size=ET.SubElement(task,'original_size');ET.SubElement(size,'width').text=str(meta['width']);ET.SubElement(size,'height').text=str(meta['height'])
    ls=ET.SubElement(task,'labels');l=ET.SubElement(ls,'label');ET.SubElement(l,'name').text='person';ET.SubElement(l,'color').text='#23a788';ET.SubElement(l,'type').text='rectangle'
    attrs=ET.SubElement(l,'attributes')
    for name in ATTRIBUTES:
        at=ET.SubElement(attrs,'attribute')
        for k,v in [('name',name),('mutable','True'),('input_type','text'),('default_value',''),('values','')]:ET.SubElement(at,k).text=v
    status={n:'automatic' for n in engine.nodes};status.update({l['b']:l['kind'] for l in result['links']})
    for t in result['tracks']:status[t['nodes'][0]]='seed_unreviewed'
    expected={};mot=[];csvrows=[];outside_count=0;clipped=0
    def emit(track,node,frame,outside,anchor):
        nonlocal clipped
        box=node['box'];b=[max(0,min(meta['width'],box[0])),max(0,min(meta['height'],box[1])),max(0,min(meta['width'],box[2])),max(0,min(meta['height'],box[3]))]
        if b[2]<=b[0] or b[3]<=b[1]:raise ValueError('Cannot export box with no image overlap')
        if not outside:clipped+=b!=box
        at={'frame':str(frame),'outside':str(int(outside)),'occluded':'0','keyframe':'1','z_order':'0',**{k:f'{v:.6f}' for k,v in zip(('xtl','ytl','xbr','ybr'),b)}}
        el=ET.SubElement(track,'box',at)
        for name,value in zip(ATTRIBUTES,[node['score'],'outside' if outside else status[node['key']],node['key'],anchor]):ET.SubElement(el,'attribute',name=name).text=str(value)
        return b
    for t in result['tracks']:
        tid=t['id'];el=ET.SubElement(root,'track',id=str(tid),label='person',source='auto');prev=None;expected[tid]={}
        for name in t['nodes']:
            node=engine.nodes[name];f=node['frame']
            if prev and f>prev['frame']+1:emit(el,prev,prev['frame']+1,True,t['anchor']);outside_count+=1
            b=emit(el,node,f,False,t['anchor']);expected[tid][f]=b
            mot.append([f+1,tid,round(b[0],4),round(b[1],4),round(b[2]-b[0],4),round(b[3]-b[1],4),node['score'],-1,-1,-1])
            csvrows.append({'frame_0based':f,'track_id':tid,'node_key':name,'association_status':status[name],'box_clipped':b!=node['box']})
            prev=node
        if prev and prev['frame']+1<meta['frames']:emit(el,prev,prev['frame']+1,True,t['anchor']);outside_count+=1
    ET.indent(root,space='  ');ET.ElementTree(root).write(dest/'annotations.xml',encoding='utf-8',xml_declaration=True)
    # Independently expand imported-style tracks: no invented boxes in gaps or after track end.
    visible=0
    for track in ET.parse(dest/'annotations.xml').getroot().findall('track'):
        tid=int(track.attrib['id']);keys=[{'frame':int(b.attrib['frame']),'outside':b.attrib['outside']=='1','box':[float(b.attrib[k]) for k in ('xtl','ytl','xbr','ybr')]} for b in track.findall('box')]
        actual=expand_track(keys,meta['frames']-1)
        if set(actual)!=set(expected[tid]):raise AssertionError('CVAT interpolation generated/dropped boxes')
        for f,b in actual.items():
            if max(abs(x-y) for x,y in zip(b,expected[tid][f]))>1e-5:raise AssertionError('CVAT box coordinates changed')
        visible+=len(actual)
    if visible!=result['stats']['visible_boxes']:raise AssertionError('Visible count mismatch')
    with (dest/'tracks.mot.txt').open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(sorted(mot))
    with (dest/'provenance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['frame_0based','track_id','node_key','association_status','box_clipped']);w.writeheader();w.writerows(csvrows)
    spec=[{'name':'person','color':'#23a788','type':'rectangle','attributes':[{'name':n,'input_type':'text','mutable':True,'default_value':'','values':['']} for n in ATTRIBUTES]}]
    (dest/'labels.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'corrections.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'segment_reviews.json').write_text(json.dumps(reviews or [],ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'duplicate_aliases.json').write_text(json.dumps(result['duplicates'],ensure_ascii=False,indent=2),encoding='utf-8')
    report={'revision':revision,'stats':result['stats'],'visible_boxes_verified':visible,'outside_markers':outside_count,
            'export_clipped_boxes':clipped,'identity_constraints_verified':True,'no_gap_interpolation_verified':True,
            'video':meta['video'],'input_sha256':hashlib.sha256((Path(directory)/'input.json').read_bytes()).hexdigest(),
            'engine_sha256':hashlib.sha256(Path(__file__).with_name('engine.py').read_bytes()).hexdigest(),
            'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'note':'Boxes are detector outputs. human_direct/transitive describe incoming identity links only, never whole-track or box ground truth. ID anchors may change after split/merge. No training.'}
    (dest/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return str(dest)
