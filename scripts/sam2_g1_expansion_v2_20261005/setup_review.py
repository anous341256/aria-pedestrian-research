"""Create this version's review implementation without modifying earlier work."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
here=Path(__file__).parent
old=(ROOT/'scripts/sam2_g1_expansion_20261005/final_review.py').read_text(encoding='utf-8')
body=old[old.index('def archive(rows):'):]
body=body.replace("n.write_json(dest/'review.json'", "n.write_json(dest/'finish_review.json'")
body=body.replace("accepted=q['valid'] and float(ss[j])>=.45 and overlap<.15 and posok", "accepted=q['valid'] and float(ss[j])>=.45 and overlap<.15 and posok and (q['box'][3]-q['box'][1]) >= .80*(y2-y1)")
# A revoked donor can have downstream recipients; quarantine until stable.
start=body.index('   invalidkeys=')
end=body.index('   for e in census',start)
body=body[:start]+'''   while True:
    invalidkeys=set()
    for key,d in logs.items():
     if key not in rows:continue
     for pt in d.get('negative_points',[]):
      donor=rows.get((pt['donor_id'],pt['frame']))
      x,y=map(int,pt['xy'])
      if donor is None or not n.decode(donor['rle'])[y,x]:invalidkeys.add(key)
    if not invalidkeys:break
    for key in invalidkeys:
     rows.pop(key,None);logs[key].update(status='REVIEW_PENDING_DONOR',reason='Final donor unavailable or changed; conservatively quarantine recipient')
     events.append(dict(action='quarantine_donor',id=key[0],frame=key[1]))
'''+body[end:]
header='''"""Versioned offline review corrections; raw V6 inference remains preserved."""
from run import near as n, OUT, ROOT
import argparse,collections,copy,time
from review_config import REVOKE,REPAIR,EXCLUDE
CLIPS=[f'{i:02d}' for i in range(22,32)]
'''
target=here/'final_review.py'
assert not target.exists()
target.write_text(header+body,encoding='utf-8')
