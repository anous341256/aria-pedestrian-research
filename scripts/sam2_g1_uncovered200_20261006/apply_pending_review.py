"""Apply explicit visual rejects only to clips whose SAM processing has not begun."""
from census import *
import shutil,datetime
d=read(OUT/'post_review.json')
for cid,rr in d['reject_proposals'].items():
 folder=OUT/cid
 if (folder/'segments').exists() or (folder/'complete.json').exists():continue
 p=folder/'census.json';c=read(p);remove=[e for e in c['segments'] if any(str(pid) in rr for pid in e['proposal_ids'])]
 if not remove:continue
 backup=folder/'census_before_temporal_review.json'
 if not backup.exists():shutil.copy2(p,backup)
 c['segments']=[e for e in c['segments'] if e not in remove];c['people']=len({e['id'] for e in c['segments']})
 c.setdefault('explicit_pre_sam_rejects',[]).extend([dict(segment=e['segment'],proposal_ids=e['proposal_ids'],reasons=[rr[str(pid)] for pid in e['proposal_ids'] if str(pid) in rr]) for e in remove])
 write(p,c);print(cid,'pre-SAM reviewed exclusions',len(remove),flush=True)
