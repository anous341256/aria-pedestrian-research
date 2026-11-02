"""Create versioned reuse of frozen A–F experiment implementations."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;OLD=ROOT/'scripts/transformer_fusion_20261005'
for name in ['network.py','shared.py','run.py','verify.py','geometry_audit.py']:
 p=HERE/name
 assert not p.exists(),p
 text=(OLD/name).read_text(encoding='utf8')
 if name=='shared.py':
  text=text.replace("OUT = ROOT/'outputs/transformer_fusion_20261005'","OUT = ROOT/'outputs/transformer_expanded_20261005'\nBASELINE = ROOT/'outputs/transformer_fusion_20261005'\nDATA = OUT/'data'\nORIGINAL_CAMERA = CAMERA\nCAMERA = OUT/'camera'")
  text=text.replace("REPORT = ROOT/'reports/transformer_fusion_20261005'","REPORT = ROOT/'reports/transformer_expanded_20261005'")
 if name in ['run.py','verify.py','geometry_audit.py']:
  text=text.replace("np.load(OLD/'dataset.npz')","np.load(DATA/'dataset.npz')").replace("read(OLD/'samples.json')","read(DATA/'samples.json')")
  text=text.replace('models=108,reused_camera_models=18','models=198,reused_camera_models=18,new_camera_models=15')
  text=text.replace('predictions_replayed=108*3','predictions_replayed=len(checks)*3').replace('Verified 108 models, 324 prediction files','Verified 198 models, 594 prediction files')
  text=text.replace('Train all 108 models','Train all 198 models')
 if name=='network.py':shutil.copyfile(OLD/name,p)
 else:p.write_text(text,encoding='utf8')
print('Versioned implementation created; network copied exactly.')
