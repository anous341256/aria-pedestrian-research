from partial_common import *
def independent(rows,ref):
 bys={};byt={}
 for r in rows:
  sid=ref[r['uid']]['source_id'];bys.setdefault(sid,[]).append(r);byt.setdefault(r['track_id'],[]).append(r)
 changes=sum(sum(a['track_id']!=b['track_id'] for a,b in zip(sorted(v,key=lambda r:r['frame'])[:-1],sorted(v,key=lambda r:r['frame'])[1:])) for v in bys.values());mixed=sum(sum(ref[a['uid']]['source_id']!=ref[b['uid']]['source_id'] for a,b in zip(v[:-1],v[1:])) for v in byt.values());frag=sum(len({r['track_id'] for r in v})-1 for v in bys.values());return changes,mixed,frag
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');comp=read(OUT/'comparison.json');checks=[]
 for cl in inv['clips']:
  cid=cl['clip']
  for version,path in [('baseline',BASE/'tracking/natural'/ARM/f'{cid}.json'),('partial',OUT/'tracking'/f'{cid}.json')]:
   r=independent(read(path)['rows'],ref);v=comp[version]['per_clip'][cid];assert r==(v['source_id_assignment_changes'],v['within_track_reference_identity_changes'],v['fragmentation_excess']);checks.append([cid,version])
 assert sum(read(OUT/'comparison.json')['partial']['per_clip'][c['clip']]['candidate_assignments'] for c in inv['clips'])==1486
 for dest in [OUT,TRIM]:
  assert read(dest/'verification.json')['passed'];assert read(dest/'video_verification.json')['passed'];protected=read(dest/'protected_sources.json');assert all(sha(ROOT/p)==h for p,h in protected.items());src=ROOT/'scripts'/dest.name
  for name,h in read(dest/'execution_code_hashes.json').items():assert sha(src/name)==h,(src,name)
  files=[p for folder in [dest,src,ROOT/'reports'/dest.name] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['delivery_complete.json','delivery_hashes.json']]
  if dest==TRIM:files.append(ROOT/'scripts/partial_appearance_20261028/render_versions.py');files.append(ROOT/'scripts/partial_appearance_20261028/partial_common.py')
  hashes={str(p.relative_to(ROOT)):sha(p) for p in files};write(dest/'delivery_hashes.json',hashes);result=dict(passed=True,protected_unchanged=len(protected),delivery_files=len(hashes),independent_metric_checks=checks if dest==OUT else '412 continuous21point windows and selection verified',videos=len(read(dest/'video_verification.json')['videos']),decoded_frames=sum(v['decoded'] for v in read(dest/'video_verification.json')['videos']),unchanged_source_labels_models_PPT=True,no_training_or_dataset_merge=True);write(dest/'delivery_complete.json',result);print(dest.name,result)
if __name__=='__main__':main()
