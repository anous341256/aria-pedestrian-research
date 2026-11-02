"""Materialize explicitly reviewed partial tracks; preserve all automatic originals."""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
from common import ROOT,OUT,CONFIG,read_json,write_json,encoder,finish_encoder,digest
sys.path.insert(0,str(ROOT/'tracking'))
from auto_annotate_people import write_cvat

def main():
    review=read_json(Path(__file__).with_name('visual_review.json'));summaries=[];all_tracks={}
    bundle=OUT/'annotation_v1';bundle.mkdir(exist_ok=True)
    composite=None
    if not (bundle/'review_60s.mp4').exists():composite=encoder(bundle/'review_60s.mp4',1280,960,10)
    raw=None
    if not (bundle/'rgb_60s.mp4').exists():raw=encoder(bundle/'rgb_60s.mp4',2560,1920,10)
    for clip_index,spec in enumerate(read_json(CONFIG)['clips']):
        clip=OUT/'clips'/spec['id'];out=clip/'annotation_v1';out.mkdir(exist_ok=True);tracks=[]
        for t in review['tracks']:
            if t['clip']!=spec['id']:continue
            boxes=[]
            for folder,key,lo,hi in t['parts']:
                p=next(p for p in read_json(clip/folder/'proposals.json')['proposals'] if p['key']==key)
                for r in p['boxes']:
                    if lo<=r['frame_id']<=hi:
                        boxes.append(dict(r,track_id=t['id'],review_status='assistant_visual_precheck',human_confirmed=False,source_proposal=key))
            assert len({r['frame_id'] for r in boxes})==len(boxes)
            tracks.append(dict(t,track_id=t['id'],boxes=boxes))
            all_tracks[100*clip_index+t['id']]=[dict(r,track_id=100*clip_index+t['id'],frame_id=r['frame_id']+clip_index*100) for r in boxes]
        payload={'spec':spec,'label_status':'assistant visually prechecked partial silver labels; NOT user-confirmed gold','scope':'selected identity intervals only; all unlisted people/frames unknown, not negative examples','tracks':tracks,'review_manifest_sha256':digest(Path(__file__).with_name('visual_review.json'))}
        write_json(out/'tracks.json',payload)
        write_cvat(out/'annotations.xml',{t['id']:t['boxes'] for t in tracks},clip/'rgb.mp4',{'width':2560,'height':1920,'fps':10,'processed_frames':100})
        if composite or raw:
            cap=cv2.VideoCapture(str(clip/'rgb.mp4'))
            byframe={f:[] for f in range(100)}
            for t in tracks:
                for r in t['boxes']:byframe[r['frame_id']].append(r)
            for f in range(100):
                ok,im=cap.read();assert ok
                if raw:raw[0].stdin.write(im.tobytes())
                if not composite:continue
                im=cv2.resize(im,(1280,960))
                for r in byframe[f]:
                    b=(np.array(r['bbox_xyxy'])/2).astype(int);tid=r['track_id'];col=[(0,230,255),(255,180,40),(220,70,220),(90,240,90)][(tid-1)%4]
                    cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),col,2);cv2.putText(im,f'P{tid}',tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,.6,col,2)
                cv2.rectangle(im,(0,0),(1280,50),(0,0,0),-1)
                cv2.putText(im,f'{spec["id"]} f={f} | PARTIAL, assistant precheck; awaiting user review',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.putText(im,'Unmarked people are UNKNOWN, not absent. Same P-number in different clips is NOT the same person.',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
                composite[0].stdin.write(im.tobytes())
            cap.release()
        summaries.append({'clip':spec['id'],'tracks':len(tracks),'boxes':sum(len(t['boxes']) for t in tracks)})
    if composite:finish_encoder(*composite)
    if raw:finish_encoder(*raw)
    write_cvat(bundle/'annotations.xml',all_tracks,bundle/'rgb_60s.mp4',{'width':2560,'height':1920,'fps':10,'processed_frames':600})
    write_json(bundle/'manifest.json',{'clips':summaries,'review_status':'assistant_precheck_not_gold','concatenation':'six clips in frozen config order, 100 frames each; no identity or forecasting across clip boundaries','review':review})
    print(summaries,flush=True)

if __name__=='__main__':main()
