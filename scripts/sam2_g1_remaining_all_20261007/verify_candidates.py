"""Independently replay candidate array provenance from retained RLE masks."""
from audit import *

def main():
 z=np.load(OUT/'dataset/strict_candidates.npz',allow_pickle=False)
 meta=read(OUT/'dataset/metadata.json');assert len(meta)==len(z['xy'])
 assert z['xy'].shape==(len(meta),21,2) and z['R'].shape==(len(meta),21,3,3)
 assert len(set(z['keys'].tolist()))==len(meta)
 cache={};count=0
 for j,r in enumerate(meta):
  cid=r['clip'];pid=r['id'];lo=r['local_start_frame']
  if cid not in cache:
   cache[cid]=({(x['id'],x['frame']):x for x in v.load(OUT/cid/'review_v2/masks_rle.jsonl.gz')},np.load(OUT/cid/'input/reference.npz'))
  rows,ref=cache[cid];assert str(z['keys'][j])==r['key'] and str(z['recordings'][j])==r['recording']
  for k in range(21):
   row=rows[pid,lo+k];mask=v.decode(row['rle']);yy,xx=np.nonzero(mask)
   box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]);center=(box[:2]+box[2:])/2
   assert np.array_equal(box,z['boxes'][j,k]),(r['key'],k,'box')
   assert np.array_equal(center,z['xy'][j,k]),(r['key'],k,'center')
   assert np.array_equal(ref['R'][lo+k],z['R'][j,k])
   assert np.array_equal(ref['position'][lo+k],z['position'][j,k])
   assert ref['times_ns'][lo+k]==z['times_ns'][j,k]
   count+=1
  assert np.array_equal(ref['K'],z['K'][j])
  assert np.max(np.abs(np.linalg.det(z['R'][j])-1))<1e-5
  assert not r['exclusion_reasons'] and r['eligible']
 result=dict(candidates=len(meta),person_frame_replays=count,exact_RLE_box_center_and_pose_match=True,unique_keys=True,independent_human_ground_truth=False)
 write(OUT/'candidate_verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
