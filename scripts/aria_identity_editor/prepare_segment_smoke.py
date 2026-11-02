"""Create a new isolated QA snapshot, including all current user feedback (read-only source)."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import time
from server import Project

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
for name in ('input.json','features.npz','manifest.json'):shutil.copyfile(args.source/name,args.output/name)
with sqlite3.connect((args.source/'corrections.sqlite3').as_uri()+'?mode=ro',uri=True) as source:
    with sqlite3.connect(args.output/'corrections.sqlite3') as target:source.backup(target)
started=time.perf_counter();project=Project(args.output);loaded=time.perf_counter()-started
started=time.perf_counter();page=project.segments.page();listed=time.perf_counter()-started
first=page['items'][0];detail=project.segments.detail(first['id'])
started=time.perf_counter();branches=project.segments.candidates(first['id'],detail['anchor']);searched=time.perf_counter()-started
report={'source_revision':project.revision,'tracks':project.result['stats']['tracks'],'old_pair_queue':len(project.result['queue']),
        'episodes':page['total'],'batch_size':len(page['items']),'first_task':detail,
        'candidate_count':len(branches['alternatives']),'load_seconds':loaded,'list_seconds':listed,'search_seconds':searched}
(args.output/'qa_prepare.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='first_task'},ensure_ascii=False))
