from associate_bulk import *

def main():
 inv=read(OUT/'inventory.json');by={c['clip']:[e for e in inv['entries'] if e['clip']==c['clip']] for c in inv['clips']};selected={};checks=[]
 for c in inv['clips']:selected.setdefault(c['recording'],c)
 for c in selected.values():
  cid=c['clip'];p,frames,feat=load_clip(c,by[cid]);saved=read(OUT/'tracking'/f'{cid}.json');z=infer(p,frames,feat);assert z==saved
  zero=LocalTracker(p['K'],p['R'][0],feat,0.);zr=[];zl=[]
  for f,obs in enumerate(frames):
   rr,ll=zero.step(obs,p['R'][f],p['c'][f],float((p['times_ns'][f]-p['times_ns'][0])/1e9),f);zr+=rr;zl.append(ll)
  baseline=read(OUT/'tracking_baseline'/f'{cid}.json');assert zr==baseline['rows'] and zl==baseline['frames']
  prefix=[[dict(o,source_id='FAKE') for o in reversed(obs)] for obs in frames[:11]];uids={o['uid'] for obs in prefix for o in obs};ff={u:feat[u] for u in uids};short={k:v[:11] if k in ['R','c','times_ns','selected_mps_ns'] else v for k,v in p.items()};trunc=infer(short,prefix,ff);assert trunc['rows']==[r for r in saved['rows'] if r['frame']<11] and trunc['frames']==saved['frames'][:11];checks.append(cid);print('PREFIX',cid,flush=True)
 meta=read(OUT/'dataset/metadata.json');d=dict(np.load(OUT/'dataset/all_continuous.npz'));lookup={t['target_key']:(e,t) for e in inv['entries'] for t in e['targets']};seen=set();occ=0;maxgeo=0.
 for i,m in enumerate(meta):
  assert len(m['uids'])==21 and len(set(m['uids']))==21 and m['frames']==list(range(21));assert tuple(m['uids']) not in seen;seen.add(tuple(m['uids']));assert len({lookup[u][0]['recording'] for u in m['uids']})==1
  for j,u in enumerate(m['uids']):
   e,t=lookup[u];assert e['frame']==j and e['rgb_timestamp_ns']==m['timestamps_ns'][j];assert np.array_equal((np.array(t['box'][:2])+np.array(t['box'][2:]))/2,d['xy'][i,j]);h=d['K'][i]@d['R'][i,15].T@d['R'][i,j]@np.linalg.inv(d['K'][i]);maxgeo=max(maxgeo,float(abs(h-d['H'][i,j]).max()));occ+=1
  assert (d['translation_mps_ns'][i]<=np.array(m['timestamps_ns'][:16])).all();assert d['seconds'][i,15]==0
 q=np.flatnonzero(d['quality_eligible']);qd=dict(np.load(OUT/'dataset/quality_subset.npz'));assert np.array_equal(qd.pop('source_indices'),q)
 for k in d:assert np.array_equal(qd[k],d[k][q])
 for f in read(OUT/'dataset/future_training_eligibility.json'):
  for i in f['all_training_eligible_indices']:assert meta[i]['recording'] not in [f['test_recording'],f['validation_recording']]
 assert maxgeo<1e-8;protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());write(OUT/'verification.json',dict(passed=True,representative_recording_prefix_replays=checks,physically_future_removed=True,order_fake_identity_invariant=True,windows_checked=len(meta),target_occurrences_checked=occ,geometry_max_difference=maxgeo,quality_subset_exact=True,recording_training_eligibility_disjoint=True,protected_files=len(protected)))
if __name__=='__main__':main()
