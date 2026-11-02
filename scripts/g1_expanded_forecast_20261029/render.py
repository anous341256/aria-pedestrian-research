from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'scripts/trimmed_forecast_finetune_20261028/render_finetune.py'
for family in ['gru','d']:
 s=source.read_text(encoding='utf-8')
 s=s.replace('from forecast_common import *',f'''import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_expanded_forecast_20261029'
BASE=ROOT/'outputs/translation_forecast_20261020'
ARMS=['{family}_original','{family}_expanded']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')''')
 s=s.replace("PRES=OUT/'presentation'",f"PRES=OUT/'presentation/{family}'")
 s=s.replace('Original-data fine-tune',family.upper()+': original data').replace('Trimmed-data fine-tune',family.upper()+': + G1 quality')
 s=s.replace('Original-data\\nfine-tune',family.upper()+': original').replace('Trimmed-data\\nfine-tune',family.upper()+': + G1')
 s=s.replace('fixed_9_trimmed_finetune_comparison.mp4',f'fixed_9_{family}_expanded.mp4').replace('Frozen camera |','Fixed test protocol |').replace('saved camera frozen','Original + G1 quality vs matched original-only')
 env=dict(__file__=__file__,__name__='__main__',Path=Path);exec(compile(s,str(source),'exec'),env)
