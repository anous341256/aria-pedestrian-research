from inspect_failures import *
def main():
 protected={**read(BASE/'source_hashes.json'),**read(OUT/'protected_sources.json')}
 assert all(sha(ROOT/p)==s for p,s in protected.items())
 replay=read(OUT/'replay_verification.json');videos=read(OUT/'video_verification.json');cases=read(OUT/'selected_cases.json');assert replay['passed'] and replay['frames']==530 and videos['passed'] and len(cases)==8
 by={c['code']:c for c in cases}
 assert by['A']['catalog_event']['counterfactual_selected_track']['remove_appearance_gate']==1
 assert by['B']['catalog_event']['counterfactual_selected_track']['remove_motion_gate']==5
 assert by['D']['catalog_event']['counterfactual_selected_track']['exclude_stale_over_0.3s']==10
 for c in cases:
  e=c['catalog_event']
  if c['type']=='mixed':
   assert e['counterfactual_selected_track']['without_depth_cost']==e['assigned']['track_id']
   assert e['counterfactual_selected_track']['same_state_rotation_only']==e['assigned']['track_id']
 q=by['F']['catalog_event']['selected_pair'];assert q['depth_standardized']>9 and q['distance_px']>540 and q['reason']=='ok'
 files=[p for folder in [ROOT/'scripts'/OUT.name,ROOT/'reports'/OUT.name,OUT] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['delivery_complete.json','delivery_hashes.json']]
 write(OUT/'delivery_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
 result=dict(passed=True,protected_unchanged=len(protected),source_frames_exact=530,selected_cases=8,videos=len(videos['checks']),decoded_frames=sum(v['decoded_frames'] for v in videos['checks']),delivery_files=len(files),source_identity_caveat='B source masks mix people; G suspected background mask drift. Original 5 cross-source alarms are not 5 human-confirmed switches.',scope='read-only original-output diagnosis; no algorithm or source annotation changes')
 write(OUT/'delivery_complete.json',result);print(result)
if __name__=='__main__':main()
