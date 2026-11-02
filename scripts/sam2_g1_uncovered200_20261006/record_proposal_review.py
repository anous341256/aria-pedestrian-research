from census import *
d=read(OUT/'initial_review.json')
reject={'001':{16:'Only legs/overlapping body fragment of existing scene identities; unresolved duplicate'},'002':{16:'Person reflection in glass, not a separate direct-view pedestrian'},'003':{11:'Railway signal, not a person'},'004':{19:'Occluding pillar/sliver; no reliable visible person mask'}}
for cid,rows in reject.items():d['reject_proposals'].setdefault(cid,{}).update({str(k):v for k,v in rows.items()})
d['merge_proposals'].update({'001':[[1,18],[7,10],[8,13]],'004':[[3,23]]})
pages=read(OUT/'refined_review/index.json')[:4]
d['temporal_reviewed_proposals']=sorted(set(d['temporal_reviewed_proposals']+[t for p in pages for t in p['tracks']]))
write(OUT/'initial_review.json',d)
