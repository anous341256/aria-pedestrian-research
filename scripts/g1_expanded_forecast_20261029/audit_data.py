from expanded_common import *

def main():
 a=dict(np.load(OUT/'expanded_dataset.npz'));old=dict(np.load(OLD/'dataset.npz'));b=dict(np.load(BULK/'dataset/quality_subset.npz'));meta=read(OUT/'expanded_samples.json');om=read(OLD/'samples.json');nm=read(BULK/'dataset/quality_metadata.json');assert meta[:117]==om
 for k,v in a.items():assert np.array_equal(v[:117],old[k])
 maxgeo=0.;maxcam=0.
 import cv2
 for i,m in enumerate(meta):
  if i>=117:
   j=m['source_quality_index'];assert m['bulk_key']==nm[j]['key'] and m['uids']==nm[j]['uids'];assert m['timestamps_ns']==nm[j]['timestamps_ns']
   for k,v in a.items():assert np.array_equal(v[i],b[k][j]),(i,k)
  assert len(m['timestamps_ns'])==21 and np.all(np.diff(m['timestamps_ns'])>0);assert np.max(abs(np.diff(m['timestamps_ns'])/1e9-.1))<.002
  for t in range(21):
   rel=a['R'][i,15].T@a['R'][i,t];h=a['K'][i]@rel@np.linalg.inv(a['K'][i]);maxgeo=max(maxgeo,float(abs(h-a['H'][i,t]).max()));maxcam=max(maxcam,float(abs(cv2.Rodrigues(a['cam'][i,t])[0]-rel).max()))
  ts=np.array(m['timestamps_ns']);assert np.allclose(a['seconds'][i],(ts-ts[15])/1e9,atol=1e-12)
  assert np.all(a['translation_mps_ns'][i]<=ts[:16])
 cf=read(OLD/'camera_folds.json');folds=read(OUT/'folds.json');plans=read(OUT/'fold_plan.json')
 for f,c,p in zip(folds,cf,plans):
  assert f['test']==c['test'] and f['validation']==c['validation'];assert f['test'] not in c['train'] and f['validation'] not in c['train'];assert p['expanded_train_indices'][:len(f['train_indices'])]==f['train_indices'];held={f['test'],f['validation']};assert all(meta[i]['recording'] not in held for i in p['expanded_train_indices']);assert {meta[i]['recording'] for i in f['test_indices']}=={f['test']};assert {meta[i]['recording'] for i in f['val_indices']}=={f['validation']}
 assert maxgeo<1e-8 and maxcam<1e-7
 write(OUT/'independent_data_audit.json',dict(passed=True,windows=len(meta),mapped_target_frames=21*len(meta),old_rows_exact=True,new_rows_exact=True,geometry_max=maxgeo,rotation_matrix_max=maxcam,original_and_camera_fold_disjoint=True,historical_pose_times_causal=True,depth_not_input='depth_reference' not in a))
 print('DATA AUDIT',len(meta),maxgeo,maxcam)

if __name__=='__main__':main()
