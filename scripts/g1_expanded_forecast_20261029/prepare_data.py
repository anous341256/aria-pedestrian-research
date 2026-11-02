from expanded_common import *

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert read(BULK/'delivery_complete.json')['passed']
 old=dict(np.load(OLD/'dataset.npz'));new=dict(np.load(BULK/'dataset/quality_subset.npz'));om=read(OLD/'samples.json');nm=read(BULK/'dataset/quality_metadata.json');assert len(om)==117 and len(nm)==299
 # Exact time-aligned, near-identical-center duplicate windows keep the old row.
 duplicates=[];keep=[]
 for j,m in enumerate(nm):
  same=[i for i,o in enumerate(om) if o['recording']==m['recording'] and o['timestamps_ns']==m['timestamps_ns'] and np.linalg.norm(old['xy'][i]-new['xy'][j],axis=-1).max()<=2.]
  if same:duplicates.append(dict(bulk_index=j,old_indices=same,key=m['key']))
  else:keep.append(j)
 keys=['xy','boxes','R','K','H','cam','seconds','translation','translation_mps_ns'];data={k:np.concatenate([old[k],new[k][keep]],axis=0) for k in keys};meta=copy.deepcopy(om)
 for j in keep:
  m=copy.deepcopy(nm[j]);m['bulk_key']=m['key'];m['key']='bulk_'+m['key'];m['source_quality_index']=j;m['source_all_index']=int(new['source_indices'][j]);m['training_source']='new_G1_quality_subset';meta.append(m)
 folds=read(OLD/'folds.json');plans=[]
 for f in folds:
  ti=f['train_indices']+[i for i in range(117,len(meta)) if meta[i]['recording'] not in [f['test'],f['validation']]]
  assert all(meta[i]['recording'] not in [f['test'],f['validation']] for i in ti)
  plans.append(dict(fold=f['fold'],test=f['test'],validation=f['validation'],original_train_count=len(f['train_indices']),expanded_train_count=len(ti),added_count=len(ti)-len(f['train_indices']),expanded_train_indices=ti,trimmed_train_indices=ti))
 cfg=dict(version=OUT.name,horizon='16history1.5s->5future0.5s',data='Original117 plus locked299 automatic-quality bulk G1; keep original labels for exact21-timestamp near-identical centers(max2px) duplicates. Same original117 evaluation only; both heldout recordings excluded in each training pool. Not merged with older412trimmed pool.',arms=['gru_original','gru_expanded','d_original','d_expanded'],gru='Matching saved with_translation camera frozen. Person starts saved matching weights; all person parameters fine-tuned240steps,batch48,AdamWlr1e-4,wd.01,clip1,noaugmentation,future+.5history flowloss; originalvalidationresidualADE every20,step0eligible;5draws10steps.',d='Previous D unchanged:2selflayers each stream,simultaneousbidirectionalcross,concatMLPdirectimage. Same frozen saved camera encoder. Trainfromscratch240steps,batch48,lr.001,wd.01,clip1,noaug,imageMSE;originalvalidationimageADEevery20,step0eligible.',sampling='Recording-balanced batches for both original and expanded. Same paired initialization and RNG schedule; actual sampled recordings/windows differ. Fixed compute, not equal epochs. Different model families are system comparisons.',features='Existing numeric inputs only. Bulk depth used for offline association, not prediction input. No new RGB,gaze,depth or3Dgeometry branch. Future reference geometry only supervises residual targets, never historical features.',seeds=[17,29,43],selection='Frozen before training; no test-based parameter/data tuning; preserve all old checkpoints and datasets; no automatic best-baseline replacement.')
 assert not (OUT/'protocol.json').exists() or read(OUT/'protocol.json')==cfg
 write(OUT/'protocol.json',cfg);write(ROOT/'plans/G1_EXPANDED_FORECAST_20261029.txt',cfg)
 np.savez_compressed(OUT/'expanded_dataset.npz',**data);write(OUT/'expanded_samples.json',meta);write(OUT/'folds.json',folds);write(OUT/'fold_plan.json',plans);write(OUT/'duplicates.json',duplicates)
 paths=[OLD/'dataset.npz',OLD/'samples.json',OLD/'folds.json',OLD/'camera_folds.json',BULK/'dataset/quality_subset.npz',BULK/'dataset/quality_metadata.json',BULK/'visual_review_notes.json',BULK/'delivery_complete.json']
 for folder in ['translation_forecast_20261020','trimmed_forecast_finetune_20261028','transformer_d_trimmed_20261029']:
  paths+=list((ROOT/'scripts'/folder).glob('*.py'))
 for f in folds:
  for seed in [17,29,43]:
   for part in ['camera','person']:paths += [OLD/'with_translation'/f'fold_{f["fold"]}/seed_{seed}/{part}/checkpoint.pt']
   paths += [OLD/'with_translation'/f'fold_{f["fold"]}/seed_{seed}/person/predictions.npz',ROOT/'outputs/trimmed_forecast_finetune_20261028/original_data_finetune'/f'fold_{f["fold"]}/seed_{seed}/predictions.npz',ROOT/'outputs/transformer_d_trimmed_20261029/original_data'/f'fold_{f["fold"]}/seed_{seed}/predictions.npz']
 write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
 for k,v in data.items():assert np.isfinite(v).all(),k
 assert np.array_equal(data['translation'][:,-1],np.zeros((len(meta),3)))
 assert all(np.all(data['translation_mps_ns'][i]<=np.array(m['timestamps_ns'][:16])) for i,m in enumerate(meta))
 assert max(abs(data['seconds'][:,15]))==0
 timesold={(m['recording'],t) for m in om for t in m['timestamps_ns']};overlap=sum(any((m['recording'],t) in timesold for t in m['timestamps_ns']) for m in nm)
 report=dict(passed=True,original_windows=117,bulk_quality_windows=299,exact_duplicates_removed=len(duplicates),expanded_windows=len(meta),expanded_recordings=len({m['recording'] for m in meta}),bulk_windows_any_time_overlap_with_old=overlap,fold_counts=[{k:p[k] for k in ['fold','original_train_count','added_count','expanded_train_count']} for p in plans],future_depth_excluded='depth_reference' not in data)
 write(OUT/'preflight.json',report);print(report,flush=True)

if __name__=='__main__':main()
