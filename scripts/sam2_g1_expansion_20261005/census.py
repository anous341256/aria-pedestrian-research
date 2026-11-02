"""Offline candidate census; all-frame proposals reused without a person cap."""
from pathlib import Path
import sys,json,math,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/sam2_g1_expansion_20261005'
SRC=ROOT/'outputs/sam2_g1_expansion_20261005'
OUT=ROOT/'outputs/sam2_g1_expansion_20261005'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)

def main():
    OUT.mkdir(exist_ok=True)
    for k in range(11,22):
        cid=f'{k:02d}';dest=OUT/cid;dest.mkdir(exist_ok=True)
        while not (OLD/cid/'proposal_tracklets.json').exists():time.sleep(2)
        tracks=read_json(OLD/cid/'proposal_tracklets.json')
        if isinstance(tracks,dict): tracks=tracks['tracklets']
        candidates=[]
        for t in tracks:
            rows=t['rows'];near=[r for r in rows if r['box'][3]-r['box'][1]>=1920*.12 and r['confidence']>=.30]
            if not near:continue
            best=max(near,key=lambda r:r['confidence']*math.sqrt(r['box'][3]-r['box'][1]))
            candidates.append(dict(proposal_id=t['proposal_id'],first=min(r['frame'] for r in rows),last=max(r['frame'] for r in rows),near_frames=[r['frame'] for r in near],anchor=best,rows=rows))
        write_json(dest/'near_proposals.json',candidates)
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            frames.append(im)
        cap.release()
        for page in range(math.ceil(len(candidates)/24)):
            sheet=Image.new('RGB',(6*245,4*295),(245,245,245));draw=ImageDraw.Draw(sheet)
            for j,t in enumerate(candidates[page*24:(page+1)*24]):
                a=t['anchor'];f=a['frame'];x1,y1,x2,y2=map(int,a['box']);pad=max(25,int((y2-y1)*.10));x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(2560,x2+pad);y2=min(1920,y2+pad)
                crop=Image.fromarray(cv2.cvtColor(frames[f][y1:y2,x1:x2],cv2.COLOR_BGR2RGB));crop.thumbnail((240,220));x=j%6*245;y=j//6*295;sheet.paste(crop,(x+(240-crop.width)//2,y));draw.text((x+3,y+222),f"{cid} P{t['proposal_id']} f{t['first']}-{t['last']}\nanchor {f} near {len(t['near_frames'])}",font=FONT,fill=(0,0,0))
            sheet.save(dest/f'census_{page+1}.jpg',quality=92)
        # Full-frame context: includes every fifth frame, not just selected targets.
        for page in range(math.ceil(len(frames)/30)):
            indices=list(range(page*30,min(len(frames),(page+1)*30),5));sheet=Image.new('RGB',(1280,3*500),(240,240,240));draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(indices):
                im=frames[f].copy()
                for t in candidates:
                    rr=next((r for r in t['rows'] if r['frame']==f),None)
                    if rr is None:continue
                    b=list(map(int,rr['box']));cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,255),3);cv2.putText(im,str(t['proposal_id']),tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
                x=j%2*640;y=j//2*500;sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(640,480)),cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+5,y+480),f'{cid} frame {f}',fill=(0,0,0),font=FONT)
            sheet.save(dest/f'context_{page+1}.jpg',quality=94)
        print(cid,'near proposals',len(candidates),flush=True)
if __name__=='__main__':main()
