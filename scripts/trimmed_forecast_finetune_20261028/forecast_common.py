from pathlib import Path
import sys,json,hashlib,copy,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/trimmed_forecast_finetune_20261028'
BASE=ROOT/'outputs/translation_forecast_20261020'
TRIM=ROOT/'outputs/track_trimming_20261028'
ASSOC=ROOT/'outputs/geometry_association_20261027'
sys.path.insert(0,str(ROOT/'scripts/translation_forecast_20261020'))
from prepare import read,write,sha,rm,rv,warp
ARMS=['original_data_finetune','trimmed_data_finetune']
