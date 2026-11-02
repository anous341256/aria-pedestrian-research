"""Sequential GPU work; log every completed model and preserve resumable results."""
from pathlib import Path
import subprocess,sys,os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/'outputs/g1_near_distribution_subset_20261006'
def main():
    for arm in ['matched','random']:
        for script in ['preflight','run','verify','evaluate']:
            env=dict(os.environ,G1_SUBSET_ARM=arm,PYTHONUNBUFFERED='1')
            print(arm,script,'START',flush=True)
            with (DEST/f'{arm}_{script}.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,str(HERE/(script+'.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(arm,script,'DONE',flush=True)
if __name__=='__main__':main()
