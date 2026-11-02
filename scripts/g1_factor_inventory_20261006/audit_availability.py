"""Audit fixed checkpoint coverage and feature support without forecast errors."""
from inventory import *

BANKS={
 'expanded_v2':ROOT/'outputs/transformer_expanded_v2_20261005',
 'matched97_new':ROOT/'outputs/transformer_matched97_20261006/new',
}
VARIANTS=['A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry']
SEEDS=[17,29,43]

def main():
 rows=read(OUT/'combined_windows.json');good=[r for r in rows if r['eligible']]
 result={}
 for name,path in BANKS.items():
  pf=read(path/'folds.json');cf={r['fold']:r for r in read(path/'camera/folds.json')}
  available={};checks=[]
  for p in pf:
   c=cf[p['fold']];test=p['test']
   valid=(test==c['test'] and test not in p['train'] and test!=p['validation'] and test not in c['train'] and test!=c['validation'])
   files=[path/f"fold_{p['fold']}/seed_{s}/{v}/checkpoint.pt" for s in SEEDS for v in VARIANTS]
   files += [path/f"camera/fold_{p['fold']}/seed_{s}/full/camera.pt" for s in SEEDS]
   missing=[str(f) for f in files if not f.is_file()]
   checks.append(dict(recording=test,fold=p['fold'],recording_disjoint=valid,expected_files=len(files),missing=missing))
   if valid and not missing:available[test]=p['fold']
  rr=[r for r in good if r['recording'] in available]
  result[name]=dict(path=str(path),fold_checks=checks,available_test_folds=available,
   summaries={co:summarize([r for r in rr if co=='combined' or (int(r['clip'])<55)==(co=='existing')],read(OUT/'thresholds_locked.json')['thresholds']) for co in ['existing','supplement','combined']},
   eligible_keys=[r['key'] for r in rr],
   unsupported_recordings=sorted({r['recording'] for r in good}-set(available)))
  print(name,{co:{k:d[k] for k in ['windows','person_nonoverlap','recording_time_nonoverlap','recordings','oa_assigned']} for co,d in result[name]['summaries'].items()})
 write(OUT/'checkpoint_availability.json',result)
 factors={}
 threshold=read(OUT/'thresholds_locked.json')['thresholds']
 for key in FACTORS:
  vals=np.array([r['features'][key] for r in good]);low=vals<=threshold[key]
  factors[key]=dict(min=float(vals.min()),p10=float(np.percentile(vals,10)),median=float(np.median(vals)),p90=float(np.percentile(vals,90)),max=float(vals.max()),low=int(low.sum()),high=int((~low).sum()))
 details=[]
 for cell in read(OUT/'combined_summary.json')['cells']:
  rr=[r for r in good if r['levels']==cell['levels']]
  details.append(dict(cell=cell['cell'],
   continuous_ranges={k:None if not rr else [min(r['features'][k] for r in rr),max(r['features'][k] for r in rr)] for k in FACTORS},
   stable_under_delete_one_DEFGH=sum(all(abs(r['features'][k]-threshold[k])>r['sensitivity'][k] for k in 'DEFGH') for r in rr),
   recordings=sorted({r['recording'] for r in rr})))
 write(OUT/'factor_ranges_and_cell_reliability.json',dict(factors=factors,cells=details,note='Delete-one history sensitivity is a conservative possible-bin-change check, not a calibrated measurement reliability rate. A/B/C are not assessed by this check.'))
 # Freeze a candidate inventory manifest, without choosing a bank using ADE.
 write(OUT/'review_manifest.json',dict(keys=[r['key'] for r in good],threshold_sha256=digest(OUT/'thresholds_locked.json'),model_bank_selected=False,forecast_errors_read=False,new_training_executed=False,notes='Feature and annotation audit only. Model support is reported separately for the two preserved banks; do not mix them.'))

if __name__=='__main__':main()
