"""Reuse verified fixed-nine renderer with explicit D-version labels/paths."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'scripts/trimmed_forecast_finetune_20261028/render_finetune.py'
s=source.read_text(encoding='utf-8')
s=s.replace('from forecast_common import *', '''import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/transformer_d_trimmed_20261029'
BASE=ROOT/'outputs/translation_forecast_20261020'
ARMS=['original_data','trimmed_data']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')''')
s=s.replace("read(OUT/'folds.json')","read(BASE/'folds.json')").replace('Original-data fine-tune','D: original data').replace('Trimmed-data fine-tune','D: trimmed data').replace('fixed_9_trimmed_finetune_comparison.mp4','fixed_9_transformer_d_comparison.mp4').replace('Frozen camera |','Frozen camera encoder |').replace('Original-data\\nfine-tune','D: original\\ndata').replace('Trimmed-data\\nfine-tune','D: trimmed\\ndata').replace('saved camera frozen','D direct image prediction')
exec(compile(s,str(source),'exec'),globals())
