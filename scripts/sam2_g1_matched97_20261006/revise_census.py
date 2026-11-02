"""Preserve first census draft; apply pre-propagation visual corrections."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_matched97_20261006'
for cid in ['33','35']:
 p=OUT/cid/'census.json';dest=OUT/cid/'census_draft1.json'
 assert not dest.exists();shutil.move(str(p),str(dest))
