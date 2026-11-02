"""Record explicit review edits from a JSON argument, preserving each revision."""
from census import *
import sys,shutil,datetime
p=OUT/'post_review.json';d=read(p);update=json.loads(sys.stdin.read());backup=OUT/'review_revisions'/f'post_{datetime.datetime.now().strftime("%H%M%S_%f")}.json';backup.parent.mkdir(exist_ok=True);shutil.copy2(p,backup)
for cid,rr in update.get('reject_proposals',{}).items():d['reject_proposals'].setdefault(cid,{}).update(rr)
d['reviewed_track_pages']=sorted(set(d['reviewed_track_pages']+update.get('pages',[])))
write(p,d)
