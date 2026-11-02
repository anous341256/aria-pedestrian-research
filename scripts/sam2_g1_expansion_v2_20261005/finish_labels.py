"""Conservatively quarantine unresolved mask overlaps; preserve review revision."""
from qa import *
import shutil
from review_config import EXCLUDE

def dump(path,items):
 with gzip.open(path,'wt',encoding='utf8') as f:
  for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')

def main():
 for cid in CLIPS:
  folder=OUT/cid/'final';dest=OUT/cid/'review_v1'
  if (folder/'finish_quarantine.json').exists():continue
  audit=read_json(OUT/cid/'qa_final/audit.json')
  dest.mkdir(exist_ok=True)
  for p in folder.iterdir():
   if p.is_file():assert not (dest/p.name).exists();shutil.copy2(p,dest/p.name)
  rr=v.load(folder/'masks_rle.jsonl.gz');dd=v.load(folder/'decisions.jsonl.gz');rows={(r['id'],r['frame']):r for r in rr};logs={(r['id'],r['frame']):r for r in dd};events=[]
  bad={(identity,c['frame']) for c in audit['conflicts'] for identity in c['ids']}
  def remove(key,reason):
   rows.pop(key,None);logs[key].update(status='REVIEW_PENDING_FINAL',reason=reason,revision=2)
   events.append(dict(id=key[0],frame=key[1],reason=reason))
  for key in sorted(bad):remove(key,'Unresolved pair-mask overlap: quarantine both candidates pending identity review')
  while True:
   invalid=set()
   for key,d in logs.items():
    if key not in rows:continue
    for pt in d.get('negative_points',[]):
     donor=rows.get((pt['donor_id'],pt['frame']));x,y=map(int,pt['xy'])
     if donor is None or not v.decode(donor['rle'])[y,x]:invalid.add(key)
   if not invalid:break
   for key in sorted(invalid):remove(key,'Negative donor revoked or changed; conservative recipient quarantine')
  dump(folder/'masks_rle.jsonl.gz',sorted(rows.values(),key=lambda r:(r['frame'],r['id'])))
  dump(folder/'decisions.jsonl.gz',sorted(logs.values(),key=lambda r:(r['frame'],r['id'])))
  census=read_json(folder/'census.json')
  for e in census['segments']:
   e['forecast_excluded_intervals']=[dict(start=a,end=b,reason=r) for a,b,r in EXCLUDE.get(cid,{}).get(e['id'],[])]
  write_json(folder/'census.json',census)
  write_json(folder/'finish_quarantine.json',dict(events=events,raw_overlap_pairs=len(audit['conflicts']),method='No mask subtraction or invented occluded pixels. Both unresolved candidates withheld; previous review revision retained.',memory_caveat='Historical raw SAM memory is not replayed; these are reviewed offline annotations.'))
  review=read_json(folder/'finish_review.json');review['final_quarantine']=events
  review['final_exclusion_review']=dict(config_sha256=digest(ROOT/'scripts'/OUT.name/'review_config.py'),intervals=EXCLUDE.get(cid,{}),note='Final temporal visual review; partial-body masks retained for review but excluded from strict forecasting windows.')
  write_json(folder/'finish_review.json',review)
  write_json(folder/'complete.json',dict(rows=len(rows),people=census['people'],segments=len(census['segments']),statuses=dict(collections.Counter(d['status'] for d in logs.values()))))
  print(cid,'quarantine',len(events),'remaining',len(rows),flush=True)

if __name__=='__main__':main()
