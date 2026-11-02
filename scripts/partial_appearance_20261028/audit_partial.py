from run_partial import *
def main():
 rng=np.random.default_rng(17);xy=rng.uniform(.15,.85,(24,2)).astype(np.float32);f=rng.normal(size=(24,384)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True);a=dict(features=f,xy=xy);idx=np.arange(12);b=dict(features=f[idx],xy=xy[idx]*.75+.08);r=local_match(a,b);assert r['valid'] and r['inliers']==12 and r['similarity']>.999
 shuffled=dict(features=f,xy=xy[rng.permutation(24)]);bad=local_match(a,shuffled);assert not bad['valid'];uniform=dict(features=np.ones((24,384),np.float32)/np.sqrt(384),xy=xy);assert not local_match(uniform,uniform)['valid']
 inv=read(BASE/'inventory.json');checks=[]
 for cl in inv['clips']:
  cid=cl['clip'];p,frames=load_clip(cid);feat=load_features(cid);saved=read(OUT/'tracking'/f'{cid}.json');actual=infer(p,frames,feat);assert actual==saved
  end=min(34,len(frames)-1);prefix=[[dict(o,source_id='FAKE',source_segment='WRONG') for o in reversed(obs)] for obs in frames[:end+1]];allowed={o['uid'] for obs in prefix for o in obs};ff={u:feat[u] for u in allowed};pp={k:(v[:end+1] if k in ['R','c','times_ns','selected_mps_ns'] else v) for k,v in p.items()};res=infer(pp,prefix,ff);assert res['rows']==[r for r in saved['rows'] if r['frame']<=end] and res['frames']==saved['frames'][:end+1] and res['pairs']==[r for r in saved['pairs'] if r['frame']<=end]
  for r in saved['rows']:
   s=r.get('local_support')
   if s:assert s['gallery_time']<=saved['frames'][r['frame']]['time'] and np.isfinite(r['appearance'])
  checks.append(cid);print('AUDIT',cid,flush=True)
 write(OUT/'verification.json',dict(passed=True,exact_replay_clips=checks,future_truncated_order_fakeID_clips=checks,synthetic_partial_correspondence=r if False else '12/12 synthetic visible points recovered',synthetic_scrambled_geometry_rejected=True,synthetic_uniform_texture_rejected=True))
if __name__=='__main__':main()
