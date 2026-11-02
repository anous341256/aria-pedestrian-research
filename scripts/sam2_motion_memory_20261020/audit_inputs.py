"""Check selected source geometry, target intervals and common seed definitions."""
import collections
import numpy as np
from selection import ROOT,OUT,read,write
def main():
 selected=read(OUT/'selection.json')
 times_by_record=collections.defaultdict(set);lengths=collections.Counter()
 result=dict(intervals=len(selected),source_frames=0,duplicate_timestamps=0,
   max_rotation_orthogonality_error=0.,max_rotation_determinant_error=0.,
   max_timestamp_gap_ms=0.,target_segment_instances=0,one_frame_segments=0,
   complete21_segments=0,max_original_batch_size=0)
 for c in selected:
  cal=np.load(ROOT/c['input']/'reference.npz')
  R=cal['R'][c['lo']:c['hi']];ts=cal['times_ns'][c['lo']:c['hi']]
  assert len(R)==21 and np.isfinite(R).all() and np.isfinite(cal['K']).all()
  assert np.all(np.diff(ts)>0)
  result['max_rotation_orthogonality_error']=max(result['max_rotation_orthogonality_error'],float(abs(np.swapaxes(R,-1,-2)@R-np.eye(3)).max()))
  result['max_rotation_determinant_error']=max(result['max_rotation_determinant_error'],float(abs(np.linalg.det(R)-1).max()))
  result['max_timestamp_gap_ms']=max(result['max_timestamp_gap_ms'],float(np.diff(ts).max()/1e6))
  for t in ts:
   if int(t) in times_by_record[c['recording']]:result['duplicate_timestamps']+=1
   times_by_record[c['recording']].add(int(t))
  result['source_frames']+=len(ts)
  batch=collections.Counter(e['batch'] for e in c['entries'])
  result['max_original_batch_size']=max(result['max_original_batch_size'],max(batch.values()))
  for e in c['entries']:
   result['target_segment_instances']+=1
   assert e['anchor_box'][3]-e['anchor_box'][1]>=.12*1920
   assert c['lo']<=e['start']<=e['anchor_frame']<=e['end']<c['hi']
   length=e['end']-e['start']+1;lengths[length]+=1
   assert len({r['frame'] for r in e['rows']})==len(e['rows'])
   assert set(range(e['start'],e['end']+1))=={r['frame'] for r in e['rows']}
 result['segment_length_distribution']=dict(sorted(lengths.items()))
 result['one_frame_segments']=lengths[1];result['complete21_segments']=lengths[21]
 assert result['duplicate_timestamps']==0
 assert result['max_rotation_orthogonality_error']<1e-5
 assert result['max_rotation_determinant_error']<1e-5
 assert result['max_original_batch_size']<=3
 write(OUT/'input_audit.json',result);print(result)
if __name__=='__main__':main()
