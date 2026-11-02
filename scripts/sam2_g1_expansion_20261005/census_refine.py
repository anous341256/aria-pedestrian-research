"""Pre-propagation temporal identity review. Preserve the first census."""
from census import *
import copy,collections
from reviewed_people import PEOPLE
REMOVE={
 ('12','N01'):[14],('12','N02'):[11,12,13],
 ('14','N05'):list(range(42,46)),
 ('15','N06'):[2,3,4],
 ('18','N04'):[2,3],
 ('21','N08'):list(range(105,111)),
}
WHY={
 '12':'Detector switches between gray cardigan and navy dress; exclude mixed boxes and do not interpolate across occlusion.',
 '14':'Flowered-shirt woman leaves; the late guide switches onto the black-shirt foreground person.',
 '15':'Rapid view change crosses a column; interpolated guide does not reliably follow the visible phone user.',
 '18':'Black-shirt boarder occludes the white-shirt person; detector follows foreground body.',
 '21':'P117 is a head crop of N06, not the later sleeveless-dress person P126.',
}
def main():
 summary=[]
 for cid in PEOPLE:
  p=OUT/cid/'census.json';backup=OUT/cid/'census_before_temporal_review.json'
  if backup.exists():continue
  backup.write_bytes(p.read_bytes());c=read_json(p);new=[];events=[];counts=collections.Counter()
  for e in sorted(c['segments'],key=lambda e:(e['id'],e['start'])):
   remove=set(REMOVE.get((cid,e['id']),[]));kept=[r for r in e['rows'] if r['frame'] not in remove]
   chunks=[]
   for r in kept:
    if not chunks or r['frame']!=chunks[-1][-1]['frame']+1:chunks.append([])
    chunks[-1].append(r)
   for rows in chunks:
    z=copy.deepcopy(e);counts[z['id']]+=1;z['segment']=f"{z['id']}_S{counts[z['id']]:02d}"
    z.update(rows=rows,start=rows[0]['frame'],end=rows[-1]['frame'])
    if z['anchor_frame'] not in [r['frame'] for r in rows]:
     a=max(rows,key=lambda r:r['confidence']*math.sqrt(max(1,r['box'][3]-r['box'][1])))
     z.update(anchor_frame=a['frame'],anchor_box=a['box'])
    if cid=='12' and z['id']=='N02' and z['start']==14:
     prop=next(t for t in read_json(OUT/cid/'proposal_tracklets.json') if t['proposal_id']==1)
     for r in z['rows']:
      rr=next(q for q in prop['rows'] if q['frame']==r['frame']);r.update(box=rr['box'],source='visually_reassigned_detector_box',proposal_id=1)
     z.update(anchor_frame=z['rows'][0]['frame'],anchor_box=z['rows'][0]['box'])
    if cid=='14' and z['id']=='N07':z['review_only_reason']='possible_cyclist_or_bicycle_interaction'
    if cid=='13' and z['id']=='N04':z['review_only_reason']='bicycle_interaction_at_boundary'
    new.append(z)
   lost=[r['frame'] for r in e['rows'] if r['frame'] in remove]
   if lost:events.append(dict(id=e['id'],removed_guide_frames=lost,reason=WHY[cid]))
  new.sort(key=lambda e:(-max(r['box'][3]-r['box'][1] for r in e['rows']),e['start']))
  for j,e in enumerate(new):e['batch']=j//3+1
  c['segments']=new;c['temporal_review_events']=events;write_json(p,c)
  summary.append(dict(clip=cid,people=c['people'],segments=len(new),requested_frames=sum(len(e['rows']) for e in new),corrections=events))
 write_json(OUT/'census_temporal_review.json',summary)
if __name__=='__main__':main()
