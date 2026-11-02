from qa import *
from window_motion import rotation_stats,eligible
from review_config import EXCLUDE
def chunks(fs):
 runs=[]
 for f in sorted(fs):
  if not runs or f!=runs[-1][-1]+1:runs.append([])
  runs[-1].append(f)
 return runs
def collect():
 allrows=[]
 for cid in CLIPS:
  folder=OUT/cid/'final'
  if not (folder/'complete.json').exists():continue
  rows={(r['id'],r['frame']):r for r in v.load(folder/'masks_rle.jsonl.gz')};logs={(d['id'],d['frame']):d for d in v.load(folder/'decisions.jsonl.gz')}
  census=read_json(folder/'census.json');cal=np.load(OUT/cid/'input/reference.npz');aa=read_json(OUT/cid/'input/alignment.json');c=[]
  for e in census['segments']:
   good=[]
   for guide in e['rows']:
    f=guide['frame'];r=rows.get((e['id'],f));d=logs.get((e['id'],f),{});ex=EXCLUDE.get(cid,{}).get(e['id'],[])
    if not r or 'UNCERTAIN' in d.get('status','') or 'PENDING' in d.get('status','') or e.get('review_only_reason'):continue
    if any(a<=f<=b for a,b,_ in ex):continue
    b=r['box'];q=r['quality']
    if b[0]<=5 or b[1]<=5 or b[2]>=2555 or b[3]>=1915 or b[3]-b[1]<.08*1920 or q['guide_iou']<.45 or q['outside']>.1:continue
    good.append(f)
   for run in chunks(good):
    for f in range(run[0],run[-1]-19,5):
     fs=list(range(f,f+21));stats=rotation_stats(cal['R'][fs],cal['times_ns'][fs]/1e9)
     if eligible(stats) and all(aa[j]['status']=='valid' and aa[j]['quality']>=.5 for j in fs):c.append(dict(clip=cid,id=e['id'],start=f,cutoff=f+5,stats=stats))
  print(cid,len(c),dict(collections.Counter(m['id'] for m in c)),flush=True);allrows+=c
 print('TOTAL',len(allrows),flush=True);return allrows
if __name__=='__main__':collect()
