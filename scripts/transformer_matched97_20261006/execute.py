"""Sequential paired run with captured logs; independent arms share exact cameras."""
from pathlib import Path
import os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/'outputs/transformer_matched97_20261006'
def call(script,arm):
 env=dict(os.environ,G1_BATCH_ARM=arm,PYTHONUNBUFFERED='1');log=DEST/f'{arm}_{script}.log'
 print(arm,script,'START',flush=True)
 with log.open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(HERE/(script+'.py'))],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(arm,script,'DONE',flush=True)
def main():
 DEST.mkdir(exist_ok=True)
 if not (DEST/'selection_locked.json').exists():call('prepare','new')
 for arm in ['old','new']:
  for script in ['camera_stage','preflight','run','verify','evaluate']:call(script,arm)
 call('compare','new')
if __name__=='__main__':main()
