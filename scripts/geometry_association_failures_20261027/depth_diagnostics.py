from render_cases import *
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');out=[]
 for case in read(OUT/'selected_cases.json'):
  cid=case['clip'];f=case['frame'];ev=case['catalog_event'];tr=read(OUT/'traces'/f'{cid}.json');entries={e['frame']:e for e in inv['entries'] if e['clip']==cid};ids={case['source_id'],case['previous_source']};hist=[]
  for t in range(max(0,case['start_frame']-8),min(len(tr),f+3)):
   e=entries[t];depth={r['target_key']:r for r in read(BASE/'depth'/e['key']/'measurements.json')} if e['targets'] else {}
   for o in e['targets']:
    uid=o['target_key'];sid=ref[uid]['source_id']
    if sid not in ids:continue
    row=next(r for r in tr[t]['rows'] if r['uid']==uid);d=depth[uid];hist.append(dict(frame=t,source_id=sid,box=o['box'],mask_area=int(decode(o['rle']).sum()),depth=d,assignment=row,prior=tr[t]['priors'].get(str(row['track_id'])),update=next((u for u in tr[t]['updates'] if u['uid']==uid),None)))
  event_priors={str(t):tr[f]['priors'].get(str(t)) for t in [ev['previous_person_track'],ev['assigned']['track_id']]}
  out.append(dict(code=case['code'],history=hist,event_priors=event_priors))
  print('\nCASE',case['code'],'assigned',ev['assigned'])
  print('PRIORS',event_priors)
  for h in hist:
   if h['frame']>=f-6:print('HISTORY',h['frame'],h['source_id'],h['mask_area'],h['depth'],h['assignment']['track_id'],h['assignment'].get('depth_update'),h['update'])
 write(OUT/'depth_and_mask_diagnostics.json',out)
if __name__=='__main__':main()
