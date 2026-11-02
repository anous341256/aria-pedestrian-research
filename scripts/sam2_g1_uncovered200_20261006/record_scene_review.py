"""Explicit scene decisions from the context sheets viewed by the assistant."""
from census import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--through',type=int,required=True);a=p.parse_args()
d=read(OUT/'initial_review.json')
d['context_reviewed_clips']=[f'{i:03d}' for i in range(1,a.through+1)]
for i in [10,11,28,29,30,31,32,33,34,35,39,41,43,51,62,63,71,72,88,90,91,92,100,101,102,103,104,107,108,111,112,113]:
 d['exclude_clips'][f'{i:03d}']='Observed stairs/escalator travel or boarding; outside the current flat-walking G1 annotation scope'
for i in [14,40]:d['exclude_clips'][f'{i:03d}']='Observed train interior with heavily occluded seated bodies; outside current G1 near-person scope'
for i in [109]:d['exclude_clips'][f'{i:03d}']='Train-door boarding/interior transition with foreground phone; outside current flat-walking scope'
for i in [45,46,47]:d['exclude_clips'][f'{i:03d}']='Map viewing: near detections are isolated arms/body parts; no eligible near pedestrian track in sampled context'
for i in [110]:d['exclude_clips'][f'{i:03d}']='Only foreground phone-screen face and holding arm; no near direct-view pedestrian'
d['proposal_source']='refined/near_proposals.json; raw proposals preserved separately'
write(OUT/'initial_review.json',d)
