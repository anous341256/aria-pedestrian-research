"""Run paired forecasting after all historical depths are available; then audit full archive."""
import sys,time,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';HERE=Path(__file__).parent
plan=json.loads((OUT/'extraction_plan.json').read_text());historical=[e for e in plan['entries'] if e['historical']]
while not all((OUT/'frames'/e['key']/'complete.json').exists() or (OUT/'frames'/e['key']/'failure.json').exists() for e in historical):
    time.sleep(10)
print('All historical frames resolved; starting paired forecast.',flush=True)
with (OUT/'forecast_execution.log').open('w') as f:
    subprocess.run([sys.executable,str(HERE/'experiment.py'),'preflight'],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(HERE/'experiment.py'),'train'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Paired forecasts finished; waiting for archive-only future depth.',flush=True)
while not (OUT/'extraction_complete.json').exists():time.sleep(10)
subprocess.run([sys.executable,str(HERE/'experiment.py'),'package'],check=True)
print('Depth archive and model comparison complete.',flush=True)
