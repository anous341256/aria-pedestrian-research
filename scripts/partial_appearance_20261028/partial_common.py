import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,collections,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/geometry_association_20261027'))
from assoc_common import read,write,sha,decode
import numpy as np
BASE=ROOT/'outputs/geometry_association_20261027'
OUT=ROOT/'outputs/partial_appearance_20261028'
TRIM=ROOT/'outputs/track_trimming_20261028'
ARM='rotation_translation_depth'

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);TRIM.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'protocol.json').exists()
 cfg=dict(version=OUT.name,cohort='Same10 clips530frames1486 candidates; all natural original full-geometry outputs are baseline. No new SAM2/depth; labels excluded from algorithms.',features='Frozen DINOv2 ViT-S/14 foreground patch tokens. Masked crop letterboxed224, mask patch occupancy>=.6; at most64 spatially uniform tokens. Original CLS retained unchanged.',local='Mutual-nearest cosine>=.55; deterministic RANSAC similarity transform threshold21/224,scale.4..2.5,rotation<=25deg; >=6inliers,>=60%mutual-pair inlier fraction,>=15%smaller-set coverage,span>=.15. Best reliable gallery support. This is geometric patch correspondence, NOT anatomical part/occlusion GT.',fusion='s=(1-w)*original_s+w*local_median_cos; w=.5*min(1,coverage/.4); w=0 if invalid. Original .45/.82 gates,.60/.30/.10 costs,geometry,retention,update rules unchanged. Raw cos scales uncalibrated; explicit experimental prior.',gallery='Initial patch entry plus latest2 reliable-match entries,>=.3s between additions; no future gallery, no source IDs. Similarity change affects later accepted memory/3D states endogenously.',validation='Zero-strength exact original replay. Exact full replay, physically truncated future feature dictionaries, candidate-order/fake-ID invariance; synthetic partial/mismatched patch geometry. No tuning after outcomes.',evaluation='Original source-ID agreement only, not human ID accuracy. Report changes,mixed-source transitions,fragments,all candidates retained,perrecording,original8 failure cases and all10 visualizations.',preservation='Separate version; original outputs/labels/checkpoints/PPT untouched. No training or data merge.')
 trim=dict(version=TRIM.name,input='ONLY original geometry_association_20261027 natural full arm, not new partial appearance arm.',method='Offline quality segmentation plus head/tail trim. Source labels never used in selection.',hard_breaks='Any frame gap. Exclude frame if largest mask component fraction<.65 or mask/box area<.15; match sim<.50,cost>.65,margin<.02,maha2>13.82,depth standardized>3; adjacent mask area ratio>2.5 in either direction breaks before current frame, not discard current.',endpoints='Then remove exactly2frames on each end of every contiguous quality segment. Further trim endpoints until occupancy>=.18,component>=.75,not at left/right 1% border,and reliable-enough matched row sim>=.55,cost<=.60,margin>=.03; birth not endpoint-ready. No interpolation/stitching.',length='Retain only >=21 continuous frames, enumerate21point windows(16history+5future); preserve short/rejected runs and all reasons. Sliding windows overlapping; also report stride21 count.',controls='Report original output-ID contiguous runs vs fixed2frame head/tail only vs quality+trim. Reduced retained errors are dataset filtering, not tracking improvement. Reference IDs only used after frozen selection for purity checks.',limits='No absolute GT; clean reference ID not sufficient to guarantee correct masks. Offline selection allowed both ends; split train/test by original recording if later used. No forecast training or dataset merge.')
 write(OUT/'protocol.json',cfg);write(TRIM/'protocol.json',trim)
 write(ROOT/'plans/PARTIAL_APPEARANCE_20261028.txt',cfg);write(ROOT/'plans/TRACK_TRIMMING_20261028.txt',trim)
 protected=dict(read(BASE/'source_hashes.json'))
 for p in list((ROOT/'scripts/geometry_association_20261027').glob('*.py'))+list((BASE/'tracking/natural'/ARM).glob('*.json'))+list((BASE/'appearance').glob('*.npz'))+[BASE/'comparison.json',BASE/'source_identity_reference.json',BASE/'protocol.json']:
  protected[str(p.relative_to(ROOT))]=sha(p)
 write(OUT/'protected_sources.json',protected);write(TRIM/'protected_sources.json',protected)
 print('LOCKED both protocols; protected',len(protected))
if __name__=='__main__':prepare()
