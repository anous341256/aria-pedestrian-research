"""Isolated experiment paths; avoid the original model's common module name."""
from pathlib import Path
import sys,json,hashlib,time,copy,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_expanded_forecast_20261029'
OLD=ROOT/'outputs/translation_forecast_20261020'
BULK=ROOT/'outputs/g1_bulk_association_20261029'
sys.path.insert(0,str(ROOT/'scripts/translation_forecast_20261020'))
from prepare import read,write,sha,warp

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def trainers():
 sys.path.insert(0,str(ROOT/'scripts/trimmed_forecast_finetune_20261028'))
 g=module('g1_gru_continuation',ROOT/'scripts/trimmed_forecast_finetune_20261028/train_finetune.py');g.OUT=OUT;g.ARMS=['gru_original','gru_expanded']
 sys.path.insert(0,str(ROOT/'scripts/transformer_d_trimmed_20261029'))
 d=module('g1_d_trial',ROOT/'scripts/transformer_d_trimmed_20261029/experiment.py');d.OUT=OUT;d.ARMS=['d_original','d_expanded']
 return g,d
