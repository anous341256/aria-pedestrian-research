"""Isolated UI smoke dataset: any test decisions must never enter the real audit."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'outputs/aria_rotation_match_review/v1'
dest=ROOT/'outputs/aria_rotation_match_review/ui_smoke'
if dest.exists():raise FileExistsError(dest)
data=json.loads((src/'dataset.json').read_text(encoding='utf-8'))
data['queries']=data['queries'][:1]
data['test_only']=True
dest.mkdir();(dest/'frames').mkdir()
blob=json.dumps(data,ensure_ascii=False).encode()
(dest/'dataset.json').write_bytes(blob)
(dest/'manifest.json').write_text(json.dumps({'dataset_sha256':hashlib.sha256(blob).hexdigest(),'test_only':True}),encoding='utf-8')
for q in data['queries']:
    for frame in (q['previous_frame'],q['frame']):
        shutil.copyfile(src/f'frames/{frame:04d}.jpg',dest/f'frames/{frame:04d}.jpg')
print(dest)
