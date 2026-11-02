"""Freeze 200 previously uncovered camera-eligible windows before person review."""
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_g1_uncovered200_20261006'
SEED=2026100623
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (OUT/'selection_locked.json').exists()
 covered=defaultdict(set);old=[];protected=[]
 roots=['sam2_near_people_20261002','sam2_g1_expansion_20261005','sam2_g1_expansion_v2_20261005','sam2_g1_matched97_20261006','sam2_g1_factor_supplement_20261006']
 for name in roots:
  root=ROOT/'outputs'/name
  if name==roots[0]:rows=[read(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{i:02d}/input/metadata.json')['source'] for i in range(1,11)]
  else:rows=read(root/'selected_clips.json');protected.append(root/'selected_clips.json')
  for row in rows:
   cid=f"{row['candidate']:02d}";mask=root/cid/'final/masks_rle.jsonl.gz'
   if not mask.exists():continue
   meta=root/cid/'input/metadata.json'
   if meta.exists():row=read(meta)['source'];protected.append(meta)
   lo=int(row.get('source_start_frame',round(row['start_s']*10)));hi=int(row.get('source_end_frame_exclusive',round(row['end_s']*10)));rec=row.get('recording_group','unknown_20260803_'+row['id'])
   covered[rec].update(range(lo,hi));old.append(dict(version=name,clip=cid,recording=rec,start_frame=lo,end_frame_exclusive=hi));protected.extend(p for p in (root/cid/'final').iterdir() if p.is_file())
 population=ROOT/'outputs/g1_population_survey_20261006/population_frame.json';protected.append(population)
 available=[r for r in read(population) if not set(range(r['start_frame'],r['start_frame']+21))&covered[r['recording']]]
 assert len(available)==500 and len(old)==63 and sum(len(v) for v in covered.values())==4720
 strata=[]
 for rec in sorted({r['recording'] for r in available}):
  rr=sorted([r for r in available if r['recording']==rec],key=lambda r:r['start_frame'])
  for j,part in enumerate(np.array_split(np.arange(len(rr)),min(4,len(rr)))):
   keys=[rr[i]['key'] for i in part];strata.append(dict(stratum=f'{rec}_Q{j+1}',keys=keys,N=len(keys),n=1))
 remaining=200-len(strata);capacity=np.array([r['N']-1 for r in strata]);ideal=remaining*capacity/capacity.sum();extra=np.floor(ideal).astype(int)
 for j in np.argsort(-(ideal-extra),kind='stable')[:remaining-int(extra.sum())]:extra[j]+=1
 rng=np.random.default_rng(SEED);bykey={r['key']:r for r in available};selected=[]
 for s,x in zip(strata,extra):
  s['n']+=int(x);s['selected']=sorted(rng.choice(s['keys'],s['n'],replace=False).tolist());s['pi']=s['n']/s['N']
  selected += [dict(bykey[k],stratum=s['stratum'],inclusion_probability=s['pi']) for k in s['selected']]
 selected.sort(key=lambda r:(r['recording'],r['start_frame']));assert len(selected)==200
 for j,r in enumerate(selected,1):r['window_id']=f'W{j:03d}'
 # Merge ONLY overlapping/touching selected frame intervals. Never bridge old labels.
 merged=[]
 for r in selected:
  start=r['start_frame'];end=start+21
  if merged and merged[-1]['recording']==r['recording'] and start<=merged[-1]['end_frame_exclusive']:
   merged[-1]['end_frame_exclusive']=max(end,merged[-1]['end_frame_exclusive']);merged[-1]['windows'].append(r['window_id'])
  else:merged.append(dict(recording=r['recording'],start_frame=start,end_frame_exclusive=end,windows=[r['window_id']],source=r))
 clips=[];maps={}
 for j,m in enumerate(merged,1):
  row=m['source'];rec=m['recording'];cid=f'{j:03d}';start=m['start_frame'];end=m['end_frame_exclusive']
  assert not set(range(start,end))&covered[rec]
  if rec not in maps:maps[rec]=list(csv.DictReader(open(row['timestamp_map'],encoding='utf-8-sig')))
  ff=[dict(x,source_frame=start+i,frame=i) for i,x in enumerate(maps[rec][start:end])];assert len(ff)==end-start
  fp=OUT/'frame_maps'/f'{cid}.json';write(fp,ff)
  clip=dict(candidate=j,clip=cid,id=row['id'],recording_group=rec,start_s=start/10,end_s=end/10,duration_s=(end-start)/10,
    source_start_frame=start,source_end_frame_exclusive=end,source_video=row['video'],source_timestamp_map=row['timestamp_map'],source_mps=row['mps'],frame_map=str(fp),window_ids=m['windows'],
    status='locked_camera_eligible_candidate_pending_person_review',scene=row['scene'])
  clips.append(clip)
  for r in selected:
   if r['window_id'] in m['windows']:r.update(clip=cid,local_start_frame=r['start_frame']-start,local_cutoff_frame=r['start_frame']-start+5)
 write(OUT/'previous_coverage.json',old);write(OUT/'available500.json',available);write(OUT/'strata.json',strata);write(OUT/'selected_windows.json',selected);write(OUT/'selected_clips.json',clips)
 spec=dict(seed=SEED,requested_windows=200,pool=500,recordings=len({r['recording'] for r in selected}),strata=len(strata),merged_clips=len(clips),unique_seconds=sum(c['duration_s'] for c in clips),
   selection='Recording by up-to-four chronological candidate-position strata; one each then proportional remaining-capacity largest-remainder allocation; SRS without replacement within strata before new RGB/person inspection. No replacement for empty or rejected scenes.',
   scope='Annotation/review only, no forecast training, no CVAT upload, no change to existing labels or models.',
   algorithm='Frozen near-person V6 SAM2.1 large bidirectional propagation, reference Aria rotations, batches of up to three; previous-batch reliable masks as negatives; automatic proposals plus explicit agent review where recorded.',
   near='12% image-height entry, 8% retention; apparent size not metric distance; no person cap; short appearances retained.',
   overlap='All 21 frames of every selected window are disjoint from all 63 existing G1 near-person annotation intervals. Windows may overlap one another; merged clips share tracking IDs.',
   labels='Provisional SAM2 labels for user review, not validated GT or guaranteed 200 positive training samples. G1 scene/target suitability recorded separately.',
   requested_unit='200 camera-eligible time windows, each 21 frames, history6/future15; a window can contain multiple people or none.')
 write(OUT/'selection_locked.json',spec)
 locked=[OUT/f for f in ['previous_coverage.json','available500.json','strata.json','selected_windows.json','selected_clips.json','selection_locked.json']]
 write(OUT/'selection_hashes.json',{p.name:sha(p) for p in locked});write(OUT/'protected_previous_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in protected})
 print(json.dumps(spec,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
