from trim_tracks import *
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');count=0;checks=[]
 for cl in inv['clips']:
  cid=cl['clip'];old=read(BASE/'tracking/natural'/ARM/f'{cid}.json')['rows'];saved=read(TRIM/'clips'/f'{cid}.json');q={a['uid']:a['quality'] for a in saved['audit']};methods,au=select(old,q);assert package(cid,methods,ref)==saved['methods'] and au==saved['audit'];fake=[dict(r,source_id='FAKE',future_prediction_error=999) for r in old];m2,_=select(fake,q)
  for name in methods:assert [[r['uid'] for r in rr] for rr in m2[name]]==[[r['uid'] for r in rr] for rr in methods[name]]
  lookup={r['uid']:r for r in old};seen=set()
  for s in saved['methods']['quality_trim']['segments']:
   assert len(s['uids'])>=21 and s['end']-s['start']+1==len(s['uids']);assert not set(s['uids'])&seen;seen.update(s['uids']);rr=[lookup[u] for u in s['uids']];assert len({r['track_id'] for r in rr})==1 and all(not frame_rule(r,q[r['uid']])[0] for r in rr);assert frame_rule(rr[0],q[rr[0]['uid']])[1] and frame_rule(rr[-1],q[rr[-1]['uid']])[1]
  for w in saved['methods']['quality_trim']['windows']:
   rr=[lookup[u] for u in w['uids']];assert len(rr)==21 and all(b['frame']==a['frame']+1 for a,b in zip(rr[:-1],rr[1:]));assert rr[15]['frame']==w['cutoff'];count+=1
  checks.append(cid)
 write(TRIM/'verification.json',dict(passed=True,exact_selection_replays=checks,fake_identity_invariance=True,windows_independently_checked=count,no_overlap_between_retained_segments=True,no_interpolated_or_cross_gap_windows=True));print('TRIM AUDIT',count)
if __name__=='__main__':main()
