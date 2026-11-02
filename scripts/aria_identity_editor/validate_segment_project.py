"""Real-data acceptance checks. Refuses non-QA projects; service must be stopped."""
import argparse
import json
from pathlib import Path
import time
from server import Project

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
if not a.directory.name.startswith('segment_smoke'):raise ValueError('QA directories only')
project=Project(a.directory);before=dict(project.labels);before_ids=dict(project.result['track_ids']);rows=[]

def task():
    project.segments.refresh()
    return next(t for t in project.segments.tasks if len(t['nodes'])>=2)

def submit(t,action,**extra):
    started=time.perf_counter()
    result=project.apply({'revision':project.revision,'task_id':t['id'],'fingerprint':t['fingerprint'],'action':action,'watched_seconds':1.2,**extra},segment=True)
    assert result['export_error'] is None
    rows.append({'action':action,'seconds':time.perf_counter()-started,'revision':project.revision})
    return result

def undo():
    project.apply({'revision':project.revision},True)
    assert project.labels==before
    assert project.result['track_ids']==before_ids

project.segments.refresh()
risks=[k for t in project.segments.tasks for k in t['risk_keys']]
assert len(risks)==len(set(risks))
assert len(project.segments.page()['items'])<=20
t=task();submit(t,'approve');undo()
t=task();oldrev=project.revision;started=time.perf_counter();c=project.segments.candidates(t['id'],t['anchor'])
rows.append({'action':'candidate_generation','seconds':time.perf_counter()-started,'count':len(c['alternatives'])})
assert project.revision==oldrev and project.labels==before
if c['alternatives']:
    submit(t,'candidate',candidate_id=c['alternatives'][0]['id']);undo()
t=task();submit(t,'defer');assert not any(k in {q for x in project.segments.page()['items'] for q in [x['id']]} for k in [t['id']])
reopened=project.segments.reopen(t['id']);assert reopened['nodes'];undo()
t=task();ns=t['nodes'];submit(t,'manual',a=ns[0],b=ns[1],value='yes');undo()
report={'passed':True,'baseline_labels_preserved':True,'baseline_ids_restored':True,'duplicate_risk_ownership':False,'timings':rows,
        'final_revision':project.revision,'pending_episodes':project.segments.metadata()['pending'],'input_pair_queue':len(project.result['queue'])}
(a.directory/'segment_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
