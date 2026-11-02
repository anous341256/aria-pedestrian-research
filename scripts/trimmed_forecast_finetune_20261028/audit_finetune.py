"""Independent NumPy/SciPy data and projection checks, no training imports."""
from forecast_common import *
from scipy.spatial.transform import Rotation

def main():
 new=dict(np.load(OUT/'trimmed_dataset.npz'));nm=read(OUT/'trimmed_samples.json');old=dict(np.load(BASE/'dataset.npz'));meta=read(BASE/'samples.json');inv=read(ASSOC/'inventory.json');lookup={t['target_key']:(e,t) for e in inv['entries'] for t in e['targets']};windows=read(TRIM/'forecast_window_candidates.json')['windows'];refs={};poses={};maxtrans=0.;maxH=0.;maxrot=0.;maxproj=0.;maxmetric=0.
 for i,m in enumerate(nm):
  cid=m['clip'];fs=m['frames'];assert len(fs)==21 and np.array_equal(np.diff(fs),np.ones(20));assert windows[i]['uids']==m['uids']
  if cid not in refs:
   e,_=lookup[m['uids'][0]];refs[cid]=dict(np.load(ROOT/e['input']/'reference.npz'));poses[cid]=dict(np.load(ASSOC/'poses'/f'{cid}.npz'))
  ref,p=refs[cid],poses[cid];K=ref['K'];R0=ref['R'][fs[15]]
  for j,(f,uid) in enumerate(zip(fs,m['uids'])):
   e,t=lookup[uid];box=np.array(t['box']);assert e['clip']==cid and e['frame']==f and e['recording']==m['recording'];assert e['rgb_timestamp_ns']==ref['times_ns'][f]
   assert np.array_equal((box[:2]+box[2:])/2,new['xy'][i,j]);assert np.array_equal(box,new['boxes'][i,j]);R=ref['R'][f];h=K@R0.T@R@np.linalg.inv(K);maxH=max(maxH,float(abs(h-new['H'][i,j]).max()));maxrot=max(maxrot,float(abs(Rotation.from_matrix(R0.T@R).as_rotvec()-new['cam'][i,j]).max()))
   if j<16:
    v=R0.T@(p['c'][f]-p['c'][fs[15]]);maxtrans=max(maxtrans,float(abs(v-new['translation'][i,j]).max()));assert p['selected_mps_ns'][f]==new['translation_mps_ns'][i,j]<=e['rgb_timestamp_ns']
  assert m['timestamp_ns']==ref['times_ns'][fs[15]];assert new['seconds'][i,15]==0
 assert maxtrans<1e-6 and maxH<1e-8 and maxrot<1e-8
 folds=read(OUT/'folds.json');plans=read(OUT/'fold_plan.json');rows=read(OUT/'metrics.json');camera_folds=read(BASE/'camera_folds.json');checked=0
 for fold,plan in zip(folds,plans):
  excluded={fold['test'],fold['validation']};assert not (set(plan['trimmed_train_recordings'])&excluded)
  expected=[i for i,m in enumerate(nm) if m['recording'] not in excluded];assert expected==plan['trimmed_train_indices'];assert camera_folds[fold['fold']]['test']==fold['test'] and camera_folds[fold['fold']]['validation']==fold['validation']
  for seed in [17,29,43]:
   src=BASE/'with_translation'/f'fold_{fold["fold"]}/seed_{seed}/person/predictions.npz'
   for arm in ['saved_translation']+ARMS:
    folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';z=dict(np.load(src if arm=='saved_translation' else folder/'predictions.npz'));assert np.array_equal(z['indices'],fold['test_indices'])
    if arm!='saved_translation':
     c=read(folder/'complete.json');assert c['training_indices']==(fold['train_indices'] if arm==ARMS[0] else expected);assert not(set(c['training_recordings'])&excluded)
    for j,i in enumerate(z['indices']):
     rot=Rotation.from_rotvec(z['rotation'][j]).as_matrix();rays=np.linalg.solve(old['K'][i],np.c_[z['component'][j],np.ones(5)].T).T;cam=np.einsum('tji,tj->ti',rot,rays);pix=(old['K'][i]@cam.T).T;pred=pix[:,:2]/pix[:,2:];error=np.linalg.norm(pred-old['xy'][i,16:21],axis=-1);maxproj=max(maxproj,float(abs(pred-z['image'][j]).max()))
     r=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['index']==i);maxmetric=max(maxmetric,abs(r['ADE']-error.mean()),abs(r['FDE']-error[-1]));checked+=1
 assert maxproj<1e-8 and maxmetric<1e-8
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());execution=read(OUT/'execution_code_hashes.json');assert all(sha(ROOT/k)==v for k,v in execution.items())
 write(OUT/'independent_audit.json',dict(passed=True,trimmed_windows_checked=len(nm),candidate_occurrences_checked=len(nm)*21,history_pose_occurrences_checked=len(nm)*16,translation_max_difference_m=maxtrans,homography_max_difference=maxH,rotation_vector_max_difference=maxrot,independent_prediction_rows=checked,scipy_projection_max_difference_px=maxproj,independent_metric_max_difference_px=maxmetric,recording_disjoint=True,original_camera_split_matches=True,protected_files=len(protected)))
 print(read(OUT/'independent_audit.json'))
if __name__=='__main__':main()
