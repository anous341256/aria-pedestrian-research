from census import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--pages',nargs='*',default=[]);p.add_argument('--quarantine',nargs='*',default=[]);p.add_argument('--strict-exclude',nargs='*',default=[]);p.add_argument('--strict-range',nargs='*',default=[]);a=p.parse_args()
path=OUT/'mask_visual_review.json';r=read(path) if path.exists() else dict(reviewed_contact_pages=[],quarantine_ids={},strict_exclude_ids={},note='Assistant sampled first/intermediate/last and weakest-guide retained masks per identity; not exhaustive human GT.')
r['reviewed_contact_pages']=sorted(set(r['reviewed_contact_pages']+[x if x.endswith('.jpg') else x+'.jpg' for x in a.pages]))
for key,items in [('quarantine_ids',a.quarantine),('strict_exclude_ids',a.strict_exclude)]:
 for x in items:
  cid,pid,reason=x.split(':',2);r[key].setdefault(cid,{})[pid]=reason
for x in a.strict_range:
 cid,pid,lo,hi,reason=x.split(':',4);r.setdefault('strict_exclude_ranges',{}).setdefault(cid,{}).setdefault(pid,[]).append(dict(start=int(lo),end=int(hi),reason=reason))
write(path,r);print('reviewed pages',len(r['reviewed_contact_pages']))
