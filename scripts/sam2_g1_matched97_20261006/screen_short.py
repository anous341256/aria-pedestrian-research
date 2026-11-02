"""Find remaining stable 21-point intervals; avoid all previously selected time."""
from screen import *
import math
from scipy.spatial.transform import Rotation
from window_motion import rotation_stats,eligible
def main():
 used=json.loads((OUT/'excluded_intervals.json').read_text('utf8'))+json.loads((OUT/'selected_clips.json').read_text('utf8'))
 results=[]
 for sid in RECORDS:
  rel,stamp,q,valid,pos=base.signals(sid);R=Rotation.from_quat(q).as_matrix();good=[]
  for f in range(0,len(R)-20,5):
   start=float(rel[f]);end=float(rel[f+20])+.1
   if any(x['id']==sid and max(start,x['start_s'])<min(end,x['end_s']) for x in used):continue
   if not valid[f:f+21].all():continue
   stats=rotation_stats(R[f:f+21],stamp[f:f+21])
   if eligible(stats):good.append((f,stats))
  clusters=[]
  for f,m in good:
   if not clusters or f-clusters[-1][-1][0]>10:clusters.append([])
   clusters[-1].append((f,m))
  chosen=[]
  for cluster in sorted(clusters,key=lambda c:-len(c)):
   start=int(math.floor(rel[cluster[0][0]]));end=min(int(rel[-1]+.1),int(math.ceil(rel[cluster[-1][0]+20]+.1)))
   if any(x['id']==sid and max(start,x['start_s'])<min(end,x['end_s']) for x in used):continue
   if end-start<3:continue
   chosen.append(dict(id=sid,start_s=start,end_s=end,eligible_grid_starts=len(cluster),metrics=base.metrics((rel,stamp,q,valid,pos),start,end)))
   if len(chosen)==3:break
  results+=chosen
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row,path in zip(results,pool.map(base.sheet,[(x['id'],x['start_s'],x['end_s']) for x in results])):row['sheet']=path
 (OUT/'screen_short_candidates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 for r in results:print(r['id'],r['start_s'],r['end_s'],'stable starts',r['eligible_grid_starts'],flush=True)
if __name__=='__main__':main()
