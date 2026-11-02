"""Side-by-side migration; source is opened read-only, destination must not exist."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.source.resolve();dest=a.output.resolve()
if source==dest or dest.exists():raise FileExistsError(dest)
dest.mkdir(parents=True)
for name in ('input.json','features.npz','manifest.json','automatic_baseline.json'):
    if (source/name).exists():shutil.copyfile(source/name,dest/name)
with sqlite3.connect((source/'corrections.sqlite3').as_uri()+'?mode=ro',uri=True) as src:
    with sqlite3.connect(dest/'corrections.sqlite3') as dst:src.backup(dst)
with sqlite3.connect(dest/'corrections.sqlite3') as c:
    rows=c.execute('SELECT pair,value,origin FROM labels ORDER BY pair').fetchall()
    revision=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
    events=c.execute('SELECT COUNT(*) FROM events').fetchone()[0]
report={'utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'source_exports':str(source/'exports'),
        'destination':str(dest),'revision':revision,'label_count':len(rows),'event_count':events,
        'labels_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest(),
        'note':'Side-by-side snapshot. Original database/exports unchanged. Continue in new editor only; later old-server edits do not synchronize.'}
(dest/'migration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
