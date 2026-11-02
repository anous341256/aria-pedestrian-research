"""Select continuous SAM2 candidates before computing any prediction errors."""
from pathlib import Path
import hashlib, json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'outputs/sam2_fullvideos_20260927'
OUT = ROOT / 'outputs/sam2_forecast_20260928'
CANDIDATES = [
    dict(key='street_S1', recording='164557', track=1, start=100, end=529, scene='Daytime street'),
    dict(key='station_S4', recording='151153', track=4, start=400, end=669, scene='Platform to escalator'),
    dict(key='dusk_follow_S1', recording='193709', track=1, start=565, end=790, scene='Dusk following'),
    dict(key='dusk_cross_S4', recording='193709', track=4, start=284, end=454, scene='Dusk crossing'),
]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    inventory=[]
    for p in sorted(SRC.glob('*/sam2_raw_v1/tracks.json')):
        for t in read(p)['tracks']:
            runs=[]
            for r in t['frames']:
                if not r.get('bbox_xyxy'): continue
                f=r['frame']
                if not runs or f!=runs[-1][-1]+1:runs.append([])
                runs[-1].append(f)
            inventory.append(dict(recording=p.parent.parent.name,track=t['id'],
                runs=[dict(start=r[0],end=r[-1],frames=len(r)) for r in sorted(runs,key=len,reverse=True)]))
    write(OUT/'candidate_inventory.json',inventory)
    for spec in CANDIDATES:
        p=SRC/spec['recording']/'sam2_raw_v1/tracks.json'
        t=next(t for t in read(p)['tracks'] if t['id']==spec['track'])
        rows=[r for r in t['frames'] if spec['start']<=r['frame']<=spec['end']]
        assert [r['frame'] for r in rows]==list(range(spec['start'],spec['end']+1))
        spec['source_annotation_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        spec['description']=t['description']
        spec['frames']=len(rows)
        spec['sampled_review_frames']=sorted(set(range(spec['start'],spec['end']+1,10))|{spec['end']})
        spec['source_annotations_user_confirmed']=False
        boxes={r['frame']:r['bbox_xyxy'] for r in rows}
        cap=cv2.VideoCapture(str(SRC/spec['recording']/'rgb.mp4'))
        tiles=[]
        for f in spec['sampled_review_frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
            b=np.array(boxes[f],dtype=int)
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(30,255,255),8)
            # Context and crop together: the crop makes identity review possible.
            context=cv2.resize(im,(384,288))
            x1,y1=np.maximum(b[:2]-100,0);x2,y2=np.minimum(b[2:]+100,[2560,1920])
            crop=im[y1:y2,x1:x2];ch,cw=crop.shape[:2];scale=min(384/cw,288/ch)
            crop=cv2.resize(crop,None,fx=scale,fy=scale)
            right=np.zeros_like(context);right[:crop.shape[0],:crop.shape[1]]=crop
            tile=np.hstack([context,right])
            cv2.rectangle(tile,(0,0),(768,25),(0,0,0),-1)
            cv2.putText(tile,f'{spec["key"]} frame {f} / {f/10:.1f}s',(5,19),0,.55,(255,255,255),1)
            tiles.append(tile)
        cap.release()
        for page in range((len(tiles)+7)//8):
            ts=tiles[page*8:(page+1)*8]
            ts += [np.zeros_like(tiles[0])]*(8-len(ts))
            sheet=np.vstack([np.hstack(ts[i:i+2]) for i in range(0,8,2)])
            p=OUT/'review'/f'{spec["key"]}_{page:02d}.jpg';p.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(p),sheet)
        print(spec['key'],len(rows),'frames',flush=True)
    write(OUT/'candidates.json',CANDIDATES)

if __name__=='__main__':main()
