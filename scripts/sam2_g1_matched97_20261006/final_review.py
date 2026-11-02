"""Versioned offline review corrections; raw V6 inference remains preserved."""
from run import near as n, OUT, ROOT
import argparse,collections,copy,time
from review_config import REVOKE,REPAIR,EXCLUDE
CLIPS=[f'{i:02d}' for i in range(32,42)]
def archive(rows):
 a={}
 for r in rows.values():a.setdefault(r['id'],{})[r['frame']]=r
 return a

def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=CLIPS);args=p.parse_args()
 with n.torch.inference_mode(),n.torch.autocast('cuda',dtype=n.torch.bfloat16):
  tracker=None;predictor=None
  for cid in args.clips:
   folder=OUT/cid;dest=folder/'final';dest.mkdir(exist_ok=True)
   if (dest/'complete.json').exists():continue
   while not (folder/'complete.json').exists():time.sleep(2)
   rows={(r['id'],r['frame']):r for r in n.inc.loadgz(folder/'masks_rle.jsonl.gz')}
   logs={(d['id'],d['frame']):d for d in n.inc.loadgz(folder/'decisions.jsonl.gz')}
   census=n.read_json(folder/'census.json');events=[]
   guides={(e['id'],r['frame']):(e,r) for e in census['segments'] for r in e['rows']}
   todo={}
   for identity,fs in REVOKE.get(cid,{}).items():
    for f in fs:
     old=rows.pop((identity,f),None);d=logs.get((identity,f),{})
     if not d:continue
     logs[identity,f]=dict(d,status='REVIEW_PENDING',revision=1,negative_points=[],memory_written=False,prior=None,reason='Visible review: wrong identity or unresolved occlusion; raw result preserved')
     events.append(dict(action='revoke',id=identity,frame=f,had_mask=old is not None))
   for identity,(fs,positive,reason) in REPAIR.get(cid,{}).items():
    for f in fs:
     if (identity,f) in guides:todo[identity,f]=(positive,reason)
   # Re-prompt direct recipients of revoked donations; never silently relabel provenance.
   for key,d in list(logs.items()):
    bad=[pt for pt in d.get('negative_points',[]) if (pt['donor_id'],pt['frame']) not in rows]
    if bad and key in rows:todo.setdefault(key,([], 'Recompute recipient of revoked negative donor'))
   frames=[]
   if todo:
    if predictor is None:
     tracker=n.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
     predictor=n.SAM2ImagePredictor(tracker._predictor)
    cap=n.cv2.VideoCapture(str(folder/'input/rgb.mp4'))
    while True:
     ok,im=cap.read()
     if not ok:break
     frames.append(im)
    cap.release()
   # Remove all known wrong masks from the donor pool before any repair.
   for key in todo:
    if key in rows:rows[key]['donor_eligible']=False
   for (identity,f),(relative_points,reason) in sorted(todo.items(),key=lambda x:(x[0][1],x[0][0])):
    e,g=guides[identity,f];box=g['box'];x1,y1,x2,y2=box
    positive=[[x1+a*(x2-x1),y1+b*(y2-y1)] for a,b in relative_points]
    donors=n.donors_at(archive(rows),f,identity,box);negative=n.inc.near_points(donors,box)
    coords=positive+[pt['xy'] for pt in negative];labels=[1]*len(positive)+[0]*len(negative)
    predictor.set_image(n.cv2.cvtColor(frames[f],n.cv2.COLOR_BGR2RGB))
    kwargs=dict(point_coords=n.np.array(coords,n.np.float32),point_labels=n.np.array(labels,n.np.int32)) if coords else {}
    mm,ss,_=predictor.predict(box=n.np.array(box),multimask_output=True,**kwargs)
    choices=[]
    for j,mask in enumerate(mm):
     m=mask.astype(bool);q=n.quality(m,box)
     overlap=max((float(n.np.count_nonzero(m&d['mask'])/max(1,min(m.sum(),d['mask'].sum()))) for d in donors),default=0.)
     posok=all(m[min(1919,int(y)),min(2559,int(x))] for x,y in positive)
     accepted=q['valid'] and float(ss[j])>=.45 and overlap<.15 and posok and (q['box'][3]-q['box'][1]) >= .80*(y2-y1)
     choices.append(dict(index=j,sam_score=float(ss[j]),quality=q,max_donor_overlap=overlap,positive_ok=posok,accepted=bool(accepted)))
    good=[c for c in choices if c['accepted']];chosen=max(good,key=lambda c:c['sam_score']+.15*c['quality']['guide_iou']) if good else None
    rows.pop((identity,f),None)
    d=dict(id=identity,frame=f,segment=e['segment'],status='REVIEW_PROMPT' if chosen else 'REVIEW_PENDING',revision=1,reason=reason,memory_written=False,prior=None,negative_points=negative,positive_points=positive,prompt_box=box,choices=choices)
    if chosen:
     q=chosen['quality'];rows[identity,f]=dict(id=identity,frame=f,segment=e['segment'],batch=e['batch'],rle=n.encode_mask(mm[chosen['index']].astype(bool)),box=q['box'],quality=q,donor_eligible=False,revision=1)
     d.update(quality=q,sam_score=chosen['sam_score'])
    logs[identity,f]=d;events.append(dict(action='prompt',id=identity,frame=f,accepted=bool(chosen),reason=reason))
    print(cid,identity,f,d['status'],flush=True)
   # Record present-frame donor validity without claiming historical memory was replayed.
   donation_audit=[]
   for key,d in logs.items():
    if key not in rows:continue
    for pt in d.get('negative_points',[]):
     donor=rows.get((pt['donor_id'],pt['frame']));valid=False
     if donor:
      x,y=map(int,pt['xy']);valid=bool(n.decode(donor['rle'])[y,x])
     donation_audit.append(dict(recipient=list(key),donor_id=pt['donor_id'],frame=pt['frame'],point=pt['xy'],inside_final_donor=valid,source_revision=pt.get('revision',0),final_revision=None if donor is None else donor.get('revision',0)))
   while True:
    invalidkeys=set()
    for key,d in logs.items():
     if key not in rows:continue
     for pt in d.get('negative_points',[]):
      donor=rows.get((pt['donor_id'],pt['frame']))
      x,y=map(int,pt['xy'])
      if donor is None or not n.decode(donor['rle'])[y,x]:invalidkeys.add(key)
    if not invalidkeys:break
    for key in invalidkeys:
     rows.pop(key,None);logs[key].update(status='REVIEW_PENDING_DONOR',reason='Final donor unavailable or changed; conservatively quarantine recipient')
     events.append(dict(action='quarantine_donor',id=key[0],frame=key[1]))
   for e in census['segments']:
    e['forecast_excluded_intervals']=[dict(start=a,end=b,reason=r) for a,b,r in EXCLUDE.get(cid,{}).get(e['id'],[])]
   n.inc.dumpgz(dest/'masks_rle.jsonl.gz',sorted(rows.values(),key=lambda r:(r['frame'],r['id'])))
   n.inc.dumpgz(dest/'decisions.jsonl.gz',sorted(logs.values(),key=lambda r:(r['frame'],r['id'])))
   n.write_json(dest/'census.json',census)
   n.write_json(dest/'finish_review.json',dict(events=events,donation_audit=donation_audit,memory_note='Raw temporal inference preserved. Corrected frames use independent image prompts; this is reviewed offline annotation, not a replayed corrected-memory tracker benchmark.',source_hashes={name:n.digest(folder/name) for name in ['census.json','masks_rle.jsonl.gz','decisions.jsonl.gz']},code_sha256=n.digest(__file__)))
   n.write_json(dest/'complete.json',dict(rows=len(rows),people=census['people'],segments=len(census['segments']),statuses=dict(collections.Counter(d['status'] for d in logs.values()))))
   del frames;n.torch.cuda.empty_cache();print(cid,'FINAL',len(rows),flush=True)
if __name__=='__main__':main()
