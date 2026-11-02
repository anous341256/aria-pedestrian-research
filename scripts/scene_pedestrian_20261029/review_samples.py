"""Deterministic scene readiness preview: first bulk clip midpoint per recording.
It is not a representative frequency estimate or verified behavior annotation.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    inv=json.loads((a.data_root/'outputs/g1_bulk_association_20261029/inventory.json').read_text())
    chosen={}
    for c in inv['clips']:chosen.setdefault(c['recording'],c)
    rows=[]; tiles=[]
    for rec,c in sorted(chosen.items()):
        entries=[e for e in inv['entries'] if e['clip']==c['clip']];e=entries[len(entries)//2]
        cap=cv2.VideoCapture(str(a.data_root/e['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,e['source_frame']);ok,im=cap.read();cap.release();assert ok
        h,w=im.shape[:2];pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pil.save(a.out/f"{c['clip']}_{rec[-6:]}.jpg")
        tile=Image.new('RGB',(480,400),'white');tile.paste(pil.resize((480,360)),(0,40));d=ImageDraw.Draw(tile)
        title=f"clip {c['clip']} / rec {rec[-6:]} / frame {e['source_frame']}"
        d.text((8,6),title,fill='black');d.text((8,23),'Scene availability only; no behavioral label',fill='black');tiles.append(tile)
        rows.append({'recording':rec,'clip':c['clip'],'input':e['input'],'source_frame':e['source_frame'],
                     'timestamp_ns':e['rgb_timestamp_ns'],'candidate_masks':len(e['targets']),
                     'source_image_size':[w,h],'image':f"{c['clip']}_{rec[-6:]}.jpg",'selection':'first bulk clip midpoint in each recording'})
    for page,start in enumerate(range(0,len(tiles),9),1):
        pageim=Image.new('RGB',(1440,1200),'#eeeeee')
        for i,im in enumerate(tiles[start:start+9]):pageim.paste(im,((i%3)*480,(i//3)*400))
        pageim.save(a.out/f'scenes_{page:02}.jpg')
    (a.out/'review_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');print('frames',len(rows))
if __name__=='__main__':main()
