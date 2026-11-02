"""Run the locked expansion, no new camera fits and no tuning after evaluation."""
from pathlib import Path
import subprocess,sys,os,time,json
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/transformer_uncovered200_20261006'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['camera_stage','preflight','run','verify','evaluate']:
        print(name,'START',flush=True)
        with (OUT/(name+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,str(HERE/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUNBUFFERED='1'),check=True)
        print(name,'DONE',flush=True)
    (OUT/'execution_complete.json').write_text(json.dumps(dict(passed=True,time=time.time())),'utf8')

if __name__=='__main__':main()
