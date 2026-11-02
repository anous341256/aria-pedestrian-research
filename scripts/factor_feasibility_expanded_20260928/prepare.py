"""Select candidate scope and review images BEFORE any forecasting errors."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/sam2_fullvideos_20260927'
OUT=ROOT/'outputs/factor_feasibility_expanded_20260928'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'candidate_review.json').exists(),'Keep frozen scope intact'
    old=read(ROOT/'outputs/factor_feasibility_20260928/candidates.json')
    inventory=[];specs=[]
    for path in sorted(SRC.glob('*/sam2_raw_v1/tracks.json')):
        sid=path.parent.parent.name
        for t in read(path)['tracks']:
            runs=[]
            for row in t['frames']:
                if not row.get('bbox_xyxy'):continue
                f=row['frame']
                if not runs or f!=runs[-1][-1]+1:runs.append([])
                runs[-1].append(f)
            qualifying=sorted([r for r in runs if len(r)>=41],key=lambda r:(-len(r),r[0]))
            inventory.append({'recording':sid,'track':t['id'],'qualifying_runs':[{'start':r[0],'end':r[-1],'frames':len(r)} for r in qualifying]})
            if not qualifying:continue
            run=qualifying[0]
            spec={'key':f'{sid}_S{t["id"]}','recording':sid,'track':t['id'],'start':run[0],'end':run[-1],
                  'frames':len(run),'description':t['description'],'source_annotation_sha256':sha(path),
                  'previous_pilot':any(x['recording']==sid and x['track']==t['id'] and x['start']==run[0] and x['end']==run[-1] for x in old),
                  'review_frames':sorted(set(range(run[0],run[-1]+1,10))|{run[-1]})}
            specs.append(spec)
            if spec['previous_pilot']:continue
            boxes={r['frame']:r['bbox_xyxy'] for r in t['frames']};cap=cv2.VideoCapture(str(SRC/sid/'rgb.mp4'));tiles=[]
            for f in spec['review_frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
                b=np.array(boxes[f],int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,230,255),7)
                context=cv2.resize(im,(384,288));x1,y1=np.maximum(b[:2]-80,0);x2,y2=np.minimum(b[2:]+80,[2560,1920])
                crop=im[y1:y2,x1:x2];h,w=crop.shape[:2];scale=min(384/w,288/h)
                crop=cv2.resize(crop,None,fx=scale,fy=scale);right=np.zeros_like(context);right[:crop.shape[0],:crop.shape[1]]=crop
                tile=np.hstack([context,right]);cv2.rectangle(tile,(0,0),(768,25),(0,0,0),-1)
                cv2.putText(tile,f'{spec["key"]} frame={f}  t={f/10:.1f}s',(8,18),0,.55,(255,255,255),1);tiles.append(tile)
            cap.release()
            for page in range((len(tiles)+7)//8):
                ts=tiles[page*8:(page+1)*8];ts += [np.zeros_like(tiles[0])]*(8-len(ts))
                sheet=np.vstack([np.hstack(ts[i:i+2]) for i in range(0,8,2)])
                p=OUT/'review'/f'{spec["key"]}_{page:02d}.jpg';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),sheet)
            print('REVIEW',spec['key'],spec['start'],spec['end'],flush=True)
    write(OUT/'candidate_inventory.json',inventory);write(OUT/'proposed_candidates.json',specs)
    write(OUT/'selection_rule.json',{'minimum_run_frames':41,'selection':'Longest continuous nonempty run per original SAM ID across ALL five already annotated videos. Ties select earlier start.',
         'review':'Before new forecast scores, inspect 1Hz and endpoints; exclude known wrong identity / duplicated target / unstable non-person mask. Do not select by improvement. Preserve pilot results and stratify new-only cases.',
         'independence':'A case is a recording-local SAM identity segment, not a unique person across videos. Same recording cases are dependent. No new data collection or SAM runs.'})
    print('CANDIDATES',len(specs),'new',sum(not s['previous_pilot'] for s in specs),flush=True)
if __name__=='__main__':main()
