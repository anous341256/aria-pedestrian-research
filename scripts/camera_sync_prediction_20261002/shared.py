"""Reuse the frozen V1 inference and geometry without editing its source files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
V1CODE=ROOT/'scripts/biflow_aria_v1_20261002'
sys.path.insert(0,str(V1CODE))
from common import read,write,sha,metrics,project_future,P,F,SCALE,gz
import numpy as np
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
CAMERA=ROOT/'outputs/camera_full_data_20261002'
OUT=ROOT/'outputs/camera_sync_prediction_20261002'
VARIANTS=['comp_future','comp_dual']
ARMS=['old_old','old_new','new_old','new_new']
LABELS={'old_old':'旧相机：原始结果','old_new':'仅重投影换新相机','new_old':'仅网络条件换新相机','new_new':'两处同步换新相机'}

