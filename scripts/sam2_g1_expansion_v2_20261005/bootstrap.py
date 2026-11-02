from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
old='sam2_g1_expansion_20261005';new=HERE.name
for name in ['prepare_input.py','discover.py','inventory.py','census.py','prepare_census.py','run.py','qa.py','export.py','render_final.py']:
 p=HERE/name;assert not p.exists()
 s=(ROOT/'scripts'/old/name).read_text('utf8').replace(old,new).replace('range(11,22)','range(22,32)')
 s=s.replace("['11','12','13','14','15','16','18','19','20','21']","[f'{i:02d}' for i in range(22,32)]")
 s=s.replace(' if i!=17','').replace("annotated_seconds=70","annotated_seconds=98")
 s=s.replace('10 new intervals, 70 seconds, after rejecting escalator candidate 17.','10 new intervals, 98 seconds; raw visual screen then rectified census review.')
 s=s.replace('All 750 frames from eleven additional selected clips','All 980 frames from ten additional selected clips')
 s=s.replace("'ALL_10_CLIPS_70s.mp4'","'ALL_10_CLIPS_98s.mp4'")
 if name=='prepare_input.py':s=s.replace("assert not (out/'rgb.mp4').exists()","if (out/'metadata.json').exists():\n            records.append(read_json(out/'metadata.json'));continue")
 p.write_text(s,encoding='utf8')
print('Versioned utility copies created; V6 core and old assets untouched.')
