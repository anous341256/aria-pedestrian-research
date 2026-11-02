"""Finish already-authorized audits and review artifacts when fitting ends."""
from pathlib import Path
import subprocess,sys,time
HERE=Path(__file__).parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs/transformer_expanded_20261005'
while not (OUT/'geometry_audit.json').exists():time.sleep(2)
for stage in ['camera_audit','figures','render','report']:
 print('STAGE',stage,flush=True);log=(OUT/f'{stage}.log').open('a',encoding='utf8')
 p=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8')
 for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
 rc=p.wait();log.close()
 if rc:raise SystemExit(f'{stage} failed {rc}')
print('All review artifacts ready.',flush=True)
