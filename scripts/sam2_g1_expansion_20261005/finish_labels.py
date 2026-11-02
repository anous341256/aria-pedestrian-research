"""Final review decisions and causal negative-point quarantine closure."""
from qa import *
import shutil

def dump(p,rows):
 with gzip.open(p,'wt',encoding='utf8') as h:
  for r in rows:h.write(json.dumps(r,ensure_ascii=False)+'\n')

def main():
 for cid in CLIPS:
  folder=OUT/cid/'final'
  if (folder/'finish_review.json').exists():continue
  shutil.copytree(folder,OUT/cid/'review_before_delivery')
  rows={(r['id'],r['frame']):r for r in v.load(folder/'masks_rle.jsonl.gz')};logs={(d['id'],d['frame']):d for d in v.load(folder/'decisions.jsonl.gz')};events=[]
  revoke={'15':[('N08',24)],'18':[('N01',4)]}.get(cid,[])
  for key in revoke:
   rows.pop(key,None);logs[key].update(status='REVIEW_PENDING_FINAL',negative_points=[],memory_written=False,reason='Final visual/overlap check: unresolved wrong or merged mask',revision=3)
   events.append(dict(action='final_revoke',id=key[0],frame=key[1]))
  # Cascade quarantines: never retain labels with invalid current-frame donor points.
  for iteration in range(len(rows)+1):
   bad=[]
   cache={}
   for key in rows:
    for pt in logs[key].get('negative_points',[]):
     dk=(pt['donor_id'],pt['frame']);donor=rows.get(dk)
     if dk not in cache:cache[dk]=v.decode(donor['rle']) if donor else None
     x,y=map(int,pt['xy'])
     if cache[dk] is None or not cache[dk][y,x]:bad.append(key);break
   if not bad:break
   for key in bad:
    rows.pop(key);logs[key].update(status='REVIEW_PENDING_DONOR',negative_points=[],memory_written=False,reason='Transitive donor quarantine',revision=3)
    events.append(dict(action='donor_quarantine',id=key[0],frame=key[1]))
  census=read_json(folder/'census.json')
  extra={('13','N01'):[(43,50,'whole-body segmentation remains unstable despite bounded repair')],('15','N01'):[(20,36,'part masks and cropped lower body despite repair')]}
  for e in census['segments']:
   for a,b,reason in extra.get((cid,e['id']),[]):e.setdefault('forecast_excluded_intervals',[]).append(dict(start=a,end=b,reason=reason))
  dump(folder/'masks_rle.jsonl.gz',sorted(rows.values(),key=lambda r:(r['frame'],r['id'])));dump(folder/'decisions.jsonl.gz',sorted(logs.values(),key=lambda d:(d['frame'],d['id'])))
  write_json(folder/'census.json',census);write_json(folder/'finish_review.json',dict(events=events,negative_point_closure_pass=True,remaining_masks=len(rows),quality_note='Sampled temporal crops, all automatic anomaly crops and contextual census checked by agent; not independent human pixel GT.'))
  write_json(folder/'complete.json',dict(rows=len(rows),people=census['people'],segments=len(census['segments']),statuses=dict(collections.Counter(d['status'] for d in logs.values()))))
  print(cid,len(rows),events,flush=True)
if __name__=='__main__':main()
