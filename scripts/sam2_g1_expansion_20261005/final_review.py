"""Versioned visible-person corrections after temporal mask review.

Full-image SAM prompts, never mask subtraction or interpolation. Failed repairs
remain pending. Old masks and memory logs are preserved in the raw revision.
"""
from run import near as n, OUT, ROOT
import argparse,collections,copy,time

CLIPS=['11','12','13','14','15','16','18','19','20','21']
# Explicitly identified identity errors or an unresolvable occlusion.
REVOKE={
 '12':{'N01':[13],'N06':[30,32,33]},
 '13':{'N02':[55]},
 '14':{'N05':list(range(1,7)),'N02':list(range(5,12)),'N08':list(range(110,114))},
 '18':{'N04':[0,1,4,5]},
}
# Relative positive points refer to reviewed guide boxes, not motion predictions.
REPAIR={
 '11':{'N05':(list(range(45,49)),[(.5,.35)],'foreground overlap at late occlusion')},
 '13':{'N01':(list(range(43,51)),[(.5,.4),(.5,.7)],'SAM retained skin/arm instead of whole visible person'),
       'N03':([6]+list(range(68,100)),[(.5,.4)],'disconnected neighboring-person pixels inflated center')},
 '15':{'N01':(list(range(17,60)),[(.5,.38),(.5,.6)],'SAM skin-only mask omitted shirt and shorts')},
 '18':{'N01':([4],[(.5,.35)],'crossing foreground mask overlap'),'N04':(list(range(12,16)),[], 'suppress foreground pink-bag identity pixels')},
 '20':{'N01':([2,3,4,5],[(.5,.4),(.5,.7)],'initial mask misses visible body')},
 '21':{'N07':(list(range(114,120)),[(.5,.15),(.5,.4),(.5,.75)],'shirt-only mask at end of short visible interval')},
}
# Visible fragments remain annotations but cannot define a stable body center.
EXCLUDE={
 '11':{'N04':[(25,34,'short/partly occluded appearance')],'N05':[(37,48,'partial occlusion')]},
 '12':{'N01':[(0,13,'crossing and exit fragments')],'N02':[(0,15,'crossing/occlusion')],'N06':[(16,33,'pillar and foreground occlusion')],'N07':[(31,49,'pillar occlusion')]},
 '13':{'N02':[(0,18,'occlusion by foreground target'),(62,68,'partial body behind foreground')],'N03':[(4,8,'occlusion')]},
 '14':{'N02':[(0,33,'foreground occlusion'),(107,127,'foreground occlusion')],'N05':[(1,6,'occlusion')],'N06':[(1,9,'occlusion')],'N08':[(110,114,'occlusion')]},
 '15':{'N07':[(20,59,'stairs and occlusion: outside clear G1 subset')],'N08':[(24,59,'stairs and occlusion: outside clear G1 subset')]},
 '18':{'N01':[(0,7,'crossing foreground occlusion'),(43,49,'wearer phone occludes lower body')],'N04':[(0,49,'persistent foreground occlusion')]},
 '19':{'N02':[(15,28,'near-person occlusion')]},
 '20':{'N02':[(18,36,'occluded body and exit fragments')]},
 '21':{'N06':[(71,83,'occlusion behind foreground white-shirt target')],'N04':[(7,19,'occlusion behind foreground striped shirt')]},
}

def archive(rows):
 a={}
 for r in rows.values():a.setdefault(r['id'],{})[r['frame']]=r
 return a

def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=CLIPS);args=p.parse_args()
 with n.torch.inference_mode(),n.torch.autocast('cuda',dtype=n.torch.bfloat16):
  tracker=n.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
  predictor=n.SAM2ImagePredictor(tracker._predictor)
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
     accepted=q['valid'] and float(ss[j])>=.45 and overlap<.15 and posok
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
   invalidkeys={tuple(a['recipient']) for a in donation_audit if not a['inside_final_donor']}
   # Conservative quarantine if a changed donor leaves a stale point after repairs.
   for key in invalidkeys:
    rows.pop(key,None);logs[key].update(status='REVIEW_PENDING_DONOR',reason='Final donor point no longer valid; quarantine, no invented provenance')
    events.append(dict(action='quarantine_donor',id=key[0],frame=key[1]))
   for e in census['segments']:
    e['forecast_excluded_intervals']=[dict(start=a,end=b,reason=r) for a,b,r in EXCLUDE.get(cid,{}).get(e['id'],[])]
   n.inc.dumpgz(dest/'masks_rle.jsonl.gz',sorted(rows.values(),key=lambda r:(r['frame'],r['id'])))
   n.inc.dumpgz(dest/'decisions.jsonl.gz',sorted(logs.values(),key=lambda r:(r['frame'],r['id'])))
   n.write_json(dest/'census.json',census)
   n.write_json(dest/'review.json',dict(events=events,donation_audit=donation_audit,memory_note='Raw temporal inference preserved. Corrected frames use independent image prompts; this is reviewed offline annotation, not a replayed corrected-memory tracker benchmark.',source_hashes={name:n.digest(folder/name) for name in ['census.json','masks_rle.jsonl.gz','decisions.jsonl.gz']},code_sha256=n.digest(__file__)))
   n.write_json(dest/'complete.json',dict(rows=len(rows),people=census['people'],segments=len(census['segments']),statuses=dict(collections.Counter(d['status'] for d in logs.values()))))
   del frames;n.torch.cuda.empty_cache();print(cid,'FINAL',len(rows),flush=True)
if __name__=='__main__':main()
