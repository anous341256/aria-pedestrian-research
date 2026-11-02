from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
old='sam2_g1_expansion_v2_20261005';new=HERE.name
for name in ['prepare_input.py','discover.py','inventory.py','census.py','temporal.py','prepare_census.py','run.py','qa.py','export.py','final_review.py','finish_labels.py','render_final.py']:
 p=HERE/name;assert not p.exists()
 s=(ROOT/'scripts'/old/name).read_text('utf8').replace(old,new).replace('range(22,32)','range(32,42)')
 s=s.replace('annotated_seconds=98','annotated_seconds=91').replace('98 seconds','91 seconds').replace('All 980 frames','All 910 frames').replace('98s','91s')
 p.write_text(s,encoding='utf8')
print('Created new utility copies; all previous versions preserved.')
