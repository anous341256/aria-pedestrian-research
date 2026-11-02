"""Cache a pinned official Mamba reference implementation; installs nothing."""
import hashlib, json, urllib.request
from pathlib import Path

OUT=Path(__file__).resolve().parent/'vendor'
OUT.mkdir(exist_ok=True)
TAG='v2.3.2.post1'
req=urllib.request.Request(f'https://api.github.com/repos/state-spaces/mamba/commits/{TAG}',headers={'User-Agent':'research-reference-check'})
with urllib.request.urlopen(req,timeout=30) as r: commit=json.load(r)['sha']
manifest={'repository':'https://github.com/state-spaces/mamba','tag':TAG,'commit':commit,'files':{}}
for remote,local in [('mamba_ssm/modules/mamba_simple.py','mamba_simple.py'),('mamba_ssm/ops/selective_scan_interface.py','selective_scan_interface.py'),('LICENSE','LICENSE')]:
    url=f'https://raw.githubusercontent.com/state-spaces/mamba/{commit}/{remote}'
    with urllib.request.urlopen(url,timeout=30) as r: data=r.read()
    (OUT/local).write_bytes(data)
    manifest['files'][local]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
