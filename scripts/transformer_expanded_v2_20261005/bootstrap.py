"""Reuse frozen implementations in a new version; never alter prior sources."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
OLD=ROOT/'scripts/transformer_expanded_20261005'
for name in ['network.py','shared.py','run.py','verify.py','geometry_audit.py','preflight.py','execute.py','evaluate.py','camera_audit.py','render.py']:
 p=HERE/name;assert not p.exists(),p
 text=(OLD/name).read_text(encoding='utf8')
 if name=='network.py':shutil.copyfile(OLD/name,p);continue
 text=text.replace("outputs/transformer_expanded_20261005'","outputs/transformer_expanded_v2_20261005'").replace("reports/transformer_expanded_20261005'","reports/transformer_expanded_v2_20261005'")
 if name=='shared.py':text=text.replace("BASELINE = ROOT/'outputs/transformer_fusion_20261005'","BASELINE = ROOT/'outputs/transformer_expanded_20261005'\nEXPANSION = ROOT/'outputs/sam2_g1_expansion_v2_20261005'")
 if name in ['run.py','verify.py']:
  text=text.replace('198','234').replace('594','702').replace('reused_camera_models=18,new_camera_models=15','reused_camera_models=33,new_camera_models=6')
 if name=='camera_audit.py':
  text=text.replace("if f['fold']<6:","if f['fold']<11:").replace("original=ORIGINAL_CAMERA/", "original=BASELINE/'camera'/").replace('if i<117','if i<199').replace('108 original','198 previous')
 if name=='evaluate.py':
  start=text.index("    subsets={'old117'");end=text.index("    summary['cohorts']",start)
  text=text[:start]+'''    subsets={'old117':lambda r:r['index']<117,'first82':lambda r:117<=r['index']<199,
             'old199':lambda r:r['index']<199,'new97':lambda r:r['index']>=199,
             'new87_nonstatic':lambda r:r['index']>=199 and not r['static_control'],
             'new_static10':lambda r:r['index']>=199 and r['static_control'],
             'static13':lambda r:r['static_control'],'all283_nonstatic':lambda r:not r['static_control']}
'''+text[end:]
  text=text.replace("r['source_version']=='old117' and r['method'] in VARIANTS","r['index']<199 and r['method'] in VARIANTS")
  text=text.replace("edge=m['edge'],height=", "edge=m['edge'],classification=m['classification'],height=")
  text=text.replace("'border':lambda r:r['edge']", "'g1_boundary':lambda r:'boundary' in r['classification'],'other_classifications':lambda r:'boundary' not in r['classification'],\n        'border':lambda r:r['edge']")
 p.write_text(text,encoding='utf8')
print('New version created; identical network, training and inference functions.')
