"""Census remaining operational camera-eligible windows, preserving exclusions."""
from pathlib import Path
import json,csv,hashlib,collections
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_g1_remaining_all_20261007'
PREV=ROOT/'outputs/sam2_g1_uncovered200_20261006'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (OUT/'selection_locked.json').exists()
 old=read(PREV/'previous_coverage.json');prior=read(PREV/'selected_clips.json')
 rejected=set(read(ROOT/'outputs/transformer_uncovered200_20261006/selection_locked.json')['user_rejection'])
 scenes=read(PREV/'initial_review.json')['exclude_clips']
 for c in prior:old.append(dict(version=PREV.name,clip=c['clip'],recording=c['recording_group'],start_frame=c['source_start_frame'],end_frame_exclusive=c['source_end_frame_exclusive'],user_rejected=c['clip'] in rejected,scene_excluded=scenes.get(c['clip'])))
 population=read(ROOT/'outputs/g1_population_survey_20261006/population_frame.json');ledger=[];selected=[]
 for r in population:
  lo=r['start_frame'];hi=lo+21;rec=r['recording'];over=[c for c in old if c['recording']==rec and c['start_frame']<hi and c['end_frame_exclusive']>lo]
  user=[c for c in over if c.get('user_rejected')];bad=[c for c in over if c.get('scene_excluded')];full=[c for c in over if c['start_frame']<=lo and c['end_frame_exclusive']>=hi]
  # A complete existing interval is already attempted, even if the scene or mask failed.
  status='prior_user_rejected_overlap' if user else 'already_processed_single_interval' if full else 'prior_scene_excluded_overlap' if bad else 'remaining_partial_context' if over else 'remaining_wholly_uncovered'
  ledger.append(dict(key=r['key'],recording=rec,start_frame=lo,status=status,prior_intervals=over))
  if status.startswith('remaining'):selected.append(dict(r,coverage_status=status,inclusion_probability=1.0))
 selected.sort(key=lambda r:(r['recording'],r['start_frame']))
 for j,r in enumerate(selected,1):r['window_id']=f'R{j:03d}'
 merged=[]
 for r in selected:
  lo=r['start_frame'];hi=lo+21
  if merged and merged[-1]['recording']==r['recording'] and lo<=merged[-1]['end'] and hi-merged[-1]['start']<=100:
   merged[-1]['end']=max(hi,merged[-1]['end']);merged[-1]['rows'].append(r)
  else:merged.append(dict(recording=r['recording'],start=lo,end=hi,rows=[r]))
 clips=[];maps={};unique=collections.defaultdict(set);new=collections.defaultdict(set);oldframes=collections.defaultdict(set)
 for c in old:oldframes[c['recording']].update(range(c['start_frame'],c['end_frame_exclusive']))
 for j,m in enumerate(merged,1):
  r=m['rows'][0];rec=m['recording'];lo=m['start'];hi=m['end'];cid=f'{j:03d}'
  if rec not in maps:maps[rec]=list(csv.DictReader(open(r['timestamp_map'],encoding='utf-8-sig')))
  fp=OUT/'frame_maps'/f'{cid}.json';write(fp,[dict(x,source_frame=lo+i,frame=i) for i,x in enumerate(maps[rec][lo:hi])])
  c=dict(candidate=j,clip=cid,id=r['id'],recording_group=rec,start_s=lo/10,end_s=hi/10,duration_s=(hi-lo)/10,source_start_frame=lo,source_end_frame_exclusive=hi,source_video=r['video'],source_timestamp_map=r['timestamp_map'],source_mps=r['mps'],frame_map=str(fp),window_ids=[x['window_id'] for x in m['rows']],status='locked_remaining_camera_candidate',scene=r['scene'])
  clips.append(c);unique[rec].update(range(lo,hi));new[rec].update(set(range(lo,hi))-oldframes[rec])
  for x in m['rows']:x.update(clip=cid,local_start_frame=x['start_frame']-lo,local_cutoff_frame=x['start_frame']-lo+5)
 spec=dict(version=OUT.name,population=835,counts=dict(collections.Counter(r['status'] for r in ledger)),selected_windows=len(selected),clips=len(clips),recordings=len(unique),encoded_frames=sum(c['source_end_frame_exclusive']-c['source_start_frame'] for c in clips),unique_frames=sum(map(len,unique.values())),new_source_frames=sum(map(len,new.values())),scope='All remaining windows of the existing 835-window camera-screened frame. Visual G1/near-target suitability assessed separately. Not all raw footage or an exhaustive person census.',selection='No sampling, replacement or forecast-error selection. All unprocessed windows except overlap with prior user rejection or recorded scene exclusion. Partial old context retained explicitly; each locked window assigned once, max100-frame clips.',algorithm='Same near-person V6/SAM2.1 large, entry12% retention8%, no total cap, batches3, bidirectional, reference rotation, reliable prior-batch current-frame negatives.',quality='Single continuous21-frame segment; retain short/uncertain tracks separately. Exclude border/uncertain masks and flagged donor dependencies from strict candidates. Sampled visual reference, not independent human GT.',authorization='Annotation and screening only; no training, active dataset merge or CVAT upload.')
 assert len(ledger)==835 and len({r['key'] for r in selected})==len(selected)
 for name,obj in [('previous_coverage.json',old),('coverage_ledger.json',ledger),('selected_windows.json',selected),('selected_clips.json',clips),('selection_locked.json',spec)]:write(OUT/name,obj)
 write(OUT/'selection_hashes.json',{p.name:sha(p) for p in OUT.glob('*.json')})
 protect=read(PREV/'protected_previous_hashes.json')
 for p in PREV.rglob('*'):
  if p.is_file() and p.suffix not in ('.mp4','.jpg','.log'):protect[p.relative_to(ROOT).as_posix()]=sha(p)
 write(OUT/'protected_previous_hashes.json',protect)
 write(OUT/'initial_review.json',dict(context_reviewed_clips=[],exclude_clips={},reject_proposals={},merge_proposals={},proposal_ranges={},temporal_reviewed_proposals=[]))
 print(json.dumps(spec,indent=2))
if __name__=='__main__':main()
