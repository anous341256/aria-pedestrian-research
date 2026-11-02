from census import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--contexts',nargs='*',default=[]);p.add_argument('--tracks',nargs='*',default=[]);p.add_argument('--reject',nargs='*',default=[]);p.add_argument('--exclude',nargs='*',default=[]);p.add_argument('--range',nargs='*',default=[]);p.add_argument('--merge',nargs='*',default=[]);a=p.parse_args()
r=read(OUT/'initial_review.json')
r['context_reviewed_clips']=sorted(set(r['context_reviewed_clips']+a.contexts))
for cid in a.tracks:
 r['temporal_reviewed_proposals']=sorted(set(r['temporal_reviewed_proposals']+[f'{cid}:{t["proposal_id"]}' for t in read(OUT/cid/'refined/near_proposals.json')]))
for x in a.reject:
 cid,pid,reason=x.split(':',2);r['reject_proposals'].setdefault(cid,{})[pid]=reason
for x in a.exclude:
 cid,reason=x.split(':',1);r['exclude_clips'][cid]=reason
for x in a.range:
 cid,pid,lo,hi=x.split(':');r['proposal_ranges'].setdefault(cid,{})[pid]=[int(lo),int(hi)]
for x in a.merge:
 cid,ids=x.split(':');r['merge_proposals'].setdefault(cid,[]).append(list(map(int,ids.split(','))))
write(OUT/'initial_review.json',r)
print('Reviewed contexts',len(r['context_reviewed_clips']),'tracks',len(r['temporal_reviewed_proposals']))
