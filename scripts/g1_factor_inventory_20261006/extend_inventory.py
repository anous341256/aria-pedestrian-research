from inventory import *
import inventory as inv
OLD_PATHS=inv.clip_paths
def new_paths(cid):
 if int(cid)<55:return OLD_PATHS(cid)
 folder=ROOT/'outputs/sam2_g1_factor_supplement_20261006'/cid
 return folder,folder/'input'
inv.clip_paths=new_paths

def main():
 source=ROOT/'outputs/sam2_g1_factor_supplement_20261006/dataset'
 meta=read(source/'all_eligible_samples.json');data=dict(np.load(source/'all_eligible_dataset.npz'))
 new=[calc(m,{k:v[i] for k,v in data.items()}) for i,m in enumerate(meta)]
 thresholds=read(OUT/'thresholds_locked.json')['thresholds']
 for r in new:
  r['levels']=[1 if r['features'][k]>thresholds[k] else -1 for k in FACTORS]
  r['visual_g1_status']='Agent temporal RGB and mask review with explicit exclusions; not independent human GT'
 old=read(OUT/'existing_windows.json');combined=old+new
 assert len({r['key'] for r in combined})==len(combined)
 write(OUT/'supplement_windows.json',new);write(OUT/'combined_windows.json',combined)
 for name,rr in [('supplement',new),('combined',combined)]:
  good=[r for r in rr if r['eligible']];s=summarize(good,thresholds)
  s['total_source_windows']=len(rr);s['excluded']=dict(collections.Counter(z for r in rr for z in r['reasons']))
  s['by_recording']={rec:dict(windows=sum(r['recording']==rec for r in good),person_nonoverlap=sum(r['recording']==rec for r in pack_nonoverlap(good)),time_nonoverlap=sum(r['recording']==rec for r in pack_nonoverlap(good,False))) for rec in sorted({r['recording'] for r in good})}
  write(OUT/f'{name}_summary.json',s)
  print(name,{k:s[k] for k in ['windows','person_nonoverlap','recording_time_nonoverlap','recordings','tracks','oa_assigned']})
  print('cells',[(r['cell'],r['windows']) for r in s['cells']])
 protection=read(OUT/'protected_sources.json');bad=[p for p,h in protection.items() if digest(ROOT/p)!=h];assert not bad,bad
 write(OUT/'integrity.json',dict(passed=True,original_files_verified=len(protection),mismatches=bad,threshold_sha256=digest(OUT/'thresholds_locked.json'),supplement_files={p.name:digest(p) for p in [source/'all_eligible_dataset.npz',source/'all_eligible_samples.json']}))
if __name__=='__main__':main()
