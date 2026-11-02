"""Frozen-candidate sequential association; provenance never enters inference."""
import json,hashlib,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/geometry_association_20261027'
VERSION=OUT.name
ARMS=['image_motion','rotation','rotation_translation_depth']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(rle):
 a=np.zeros((1920,2560),np.uint8);x1,y1,x2,y2=map(int,rle[-4:]);a[y1:y2+1,x1:x2+1]=np.repeat(np.arange(len(rle)-4)%2,np.asarray(rle[:-4],int)).reshape(y2-y1+1,x2-x1+1);return a
def inventory():
 assert not (OUT/'protocol.json').exists(),'Preserve existing protocol'
 OUT.mkdir(parents=True,exist_ok=True);entries=[];clips=[];provenance={};protected={};n=0
 for k in range(1,11):
  cid=f'{k:02d}';inp=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input';ann=ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz'
  ref=dict(np.load(inp/'reference.npz'));meta=read(inp/'metadata.json');rec=meta['source']['recording_group']
  with gzip.open(ann,'rt') as fp:rows=list(map(json.loads,fp))
  for p in [ann,inp/'reference.npz',inp/'metadata.json',inp/'rgb.mp4']:protected[str(p.relative_to(ROOT))]=sha(p)
  for f,ts in enumerate(ref['times_ns']):
   targets=[]
   for r in rows:
    if r['frame']!=f:continue
    uid=hashlib.sha256(json.dumps([cid,f,r['rle']],separators=(',',':')).encode()).hexdigest()[:24]
    assert uid not in provenance
    targets.append(dict(target_key=uid,rle=r['rle'],box=r['box']))
    provenance[uid]=dict(clip=cid,frame=f,source_id=r['id'],source_segment=r['segment'])
   targets.sort(key=lambda t:t['target_key']);key=f'{rec}_{int(ts)}'
   e=dict(key=key,clip=cid,frame=f,recording=rec,rgb_timestamp_ns=int(ts),input=str(inp.relative_to(ROOT)),targets=targets,cached_raw=None,cached_complete=None)
   if targets:
    for ver in ['full3d_forecast_20261021','depth_translation_forecast_20261021','depth_forecast_20261008']:
     p=ROOT/f'outputs/{ver}/frames/{key}/complete.json'
     if p.exists():
      comp=read(p);raw=comp.get('raw_path',str((p.parent/'depth.npz').relative_to(ROOT)))
      if (ROOT/raw).exists():e.update(cached_raw=raw,cached_complete=str(p.relative_to(ROOT)));break
   entries.append(e);n+=len(targets)
  clips.append(dict(clip=cid,recording=rec,frames=len(ref['times_ns']),candidates=len(rows),source_id_count=len({r['id'] for r in rows})))
 for p in (ROOT/'scripts/translation_forecast_20261020').glob('*.py'):protected[str(p.relative_to(ROOT))]=sha(p)
 for p in [ROOT/'outputs/translation_forecast_20261020/comparison.json',ROOT/'scripts/depth_forecast_20261008/extract.py',ROOT/'outputs/camera_full_data_20261002/calibrations.json']:protected[str(p.relative_to(ROOT))]=sha(p)
 cfg=dict(version=VERSION,arms=ARMS,cohort='All10 original near-person clips,530frames,1486 frozen final SAM2 masks; no score-based selection or person cap.',
  candidates='Reused offline bidirectional, identity-guided and reviewed SAM2 masks. Original IDs/segments excluded from inference; same anonymous candidate pool all arms. Conditional association trial, NOT online end-to-end SAM2 evaluation or detector recall.',
  geometry='Xworld=Rprev (Z K^-1 pixel)+cprev; predict pedestrian world XYZ with CV Kalman; Xcurrent=Rcurrent.T (Xworld-c_current); project K. Last-at/before-current MPS device pose plus factory RGB lever arm. No predicted future camera pose.',
  association='Frozen DINOv2 ViT-S/14 masked crops, normalized CLS embeddings; global one-to-one Hungarian assignment with unmatched option; same appearance/size/lifecycle rules all arms. Raw image CV vs rotation-stabilized2D CV vs full3D position prediction with rotation2D fallback.',
  uncertainty='Pixel observation sigma=max(10px,.04*boxheight), depth sigma=max(.05m,.05Z,IQR/1.349), plus .02m pose floor (heuristic, not calibrated). CV acceleration sigma2D200px/s2,3D3m/s2. 3D age<=.5s and relative predicted Z uncertainty<=.25, otherwise rotation fallback. Depth innovations gated; uncertain matches do not update appearance memory.',
  scoring='0.60 appearance distance +0.30 motion distance +0.10 size change; normal2D Mahalanobis gate13.82, strong appearance>=.82 may recover inside bounded spatial reach. Full geometry motion adds .20 standardized depth discrepancy when usable. No learned weights/no test tuning.',
  lifecycle='Keep unobserved track<=1.0s. Assignments separated from memory updates. Update appearance only cosine>=.70,cost<.50, row+column alternative margins>=.08. Uncertain matches use inflated measurement noise and skip3D state update. New IDs monotonically assigned; no source identity access.',
  controls='Natural candidate stream plus separately labelled synthetic all-candidate blackout frames20..22,40..42,60..62 in every clip; skipped candidate frames also withhold embeddings/depth. Same masks otherwise. No future observations or interpolation in tracker.',
  evaluation='Identity switches, fragmentation and pairwise purity relative to PRE-EXISTING SAM/review source IDs are agreement diagnostics, not human GT ID accuracy. Report frame/clip/recording aggregates, geometry coverage, rejected depth updates, raw costs and fixed all10 videos. Independently inspect helpful and harmful disagreements.',
  depth='Official sequential512 vitl32 LR1px; stereo exposures<=current RGB, left BEFORE RGB-1ms, max40ms gap, stereo skew<=1ms; min30points,LR>=.7,Z.5..30m,IQR/Z<=.25. Surface median Z on box-center ray is a proxy, not physical body center.',
  preservation='All work isolated. No source mask modifications, no forecasting retraining, no CVAT upload/PPT edits. Offline closed-loop MPS and stereo registration not proof of realtime causal pose availability.')
 write(OUT/'protocol.json',cfg);write(ROOT/'plans/GEOMETRY_ASSOCIATION_20261027.txt',cfg)
 write(OUT/'inventory.json',dict(entries=entries,clips=clips,total_frames=len(entries),total_candidates=n,cached_frames=sum(bool(e['cached_raw']) for e in entries),new_depth_frames=sum(bool(e['targets']) and not e['cached_raw'] for e in entries)))
 write(OUT/'source_identity_reference.json',provenance);write(OUT/'source_hashes.json',protected)
 print({k:v for k,v in read(OUT/'inventory.json').items() if k not in ['entries']},flush=True)
if __name__=='__main__':inventory()
