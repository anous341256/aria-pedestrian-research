from pathlib import Path
import sys,json,hashlib,gzip,time,collections,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_bulk_association_20261029';SOURCE=ROOT/'outputs/sam2_motion_memory_20261020'
sys.path.insert(0,str(ROOT/'scripts/geometry_association_20261027'))
from assoc_common import read,write,sha,decode

def inventory():
 OUT.mkdir(parents=True,exist_ok=True);selection=read(SOURCE/'selection.json');assert len(selection)==191;entries=[];clips=[];provenance={};protected={};caches={};refs={}
 for version,sub in [('geometry_association_20261027','depth'),('full3d_forecast_20261021','frames'),('depth_translation_forecast_20261021','frames'),('depth_forecast_20261008','frames')]:
  for p in (ROOT/'outputs'/version/sub).glob('*/complete.json'):
   comp=read(p);raw=ROOT/comp.get('raw_path',str((p.parent/'depth.npz').relative_to(ROOT)))
   if raw.exists() and 'timing' in comp:caches.setdefault(p.parent.name,(str(raw.relative_to(ROOT)),str(p.relative_to(ROOT))))
 oldmeta=read(ROOT/'outputs/translation_forecast_20261020/samples.json');oldtimes={(m['recording'],int(t)) for m in oldmeta for t in m['timestamps_ns']};oldgeom=read(ROOT/'outputs/geometry_association_20261027/inventory.json');oldgtimes={(e['recording'],e['rgb_timestamp_ns']) for e in oldgeom['entries']}
 for c in selection:
  cid=c['id'];inp=ROOT/c['input'];ref=dict(np.load(inp/'reference.npz'));ann=SOURCE/cid/'external_box/masks_rle.jsonl.gz'
  with gzip.open(ann,'rt') as fp:rows=[json.loads(l) for l in fp]
  for p in [ann,inp/'reference.npz',SOURCE/cid/'seeds.jsonl.gz']:protected[str(p.relative_to(ROOT))]=sha(p)
  for f in range(c['lo'],c['hi']):
   ts=int(ref['times_ns'][f]);targets=[]
   for r in rows:
    if r['frame']!=f:continue
    uid=hashlib.sha256(json.dumps([cid,f,r['rle']],separators=(',',':')).encode()).hexdigest()[:24];assert uid not in provenance;targets.append(dict(target_key=uid,rle=r['rle'],box=r['box']));provenance[uid]=dict(clip=cid,frame=f-c['lo'],source_frame=f,source_id=r['id'],source_segment=r['segment'])
   key=f'{c["recording"]}_{ts}';cached=caches.get(key);entries.append(dict(key=key,clip=cid,frame=f-c['lo'],source_frame=f,recording=c['recording'],rgb_timestamp_ns=ts,input=c['input'],targets=sorted(targets,key=lambda t:t['target_key']),cached_raw=cached[0] if cached else None,cached_complete=cached[1] if cached else None))
  clips.append(dict(clip=cid,source=c['source'],source_clip=c['clip'],input=c['input'],recording=c['recording'],lo=c['lo'],hi=c['hi'],absolute_start=c['absolute_start'],absolute_end=c['absolute_end'],frames=c['hi']-c['lo'],candidates=len(rows),camera_stats=c['camera_stats']))
 assert len({(e['recording'],e['rgb_timestamp_ns']) for e in entries})==len(entries)
 for p in [SOURCE/'selection.json',SOURCE/'forbidden_intervals.json',SOURCE/'manual_review_flags.json']:protected[str(p.relative_to(ROOT))]=sha(p)
 for folder in ['geometry_association_20261027','partial_appearance_20261028','track_trimming_20261028']:
  for p in (ROOT/'scripts'/folder).glob('*.py'):protected[str(p.relative_to(ROOT))]=sha(p)
 summary=dict(intervals=len(clips),recordings=len({c['recording'] for c in clips}),source_clips=len({(c['source'],c['source_clip']) for c in clips}),frames=len(entries),seconds=len(entries)/10,candidates=len(provenance),depth_frames=sum(bool(e['targets']) for e in entries),cached_depth_frames=sum(bool(e['targets']) and bool(e['cached_raw']) for e in entries),new_depth_frames=sum(bool(e['targets']) and not e['cached_raw'] for e in entries),frames_outside_original117=sum((e['recording'],e['rgb_timestamp_ns']) not in oldtimes for e in entries),frames_outside_original10=sum((e['recording'],e['rgb_timestamp_ns']) not in oldgtimes for e in entries),new_raw_video_frames=0)
 write(OUT/'inventory.json',dict(entries=entries,clips=clips,summary=summary));write(OUT/'source_identity_reference.json',provenance);write(OUT/'source_hashes.json',protected);write(OUT/'inventory_summary.json',summary);print(summary,flush=True)
if __name__=='__main__':inventory()
