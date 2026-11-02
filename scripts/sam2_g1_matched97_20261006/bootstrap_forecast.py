from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2];old='transformer_expanded_v2_20261005';new='transformer_matched97_20261006'
dest=ROOT/'scripts'/new;dest.mkdir(exist_ok=True)
for name in ['network.py','run.py','shared.py']:
 p=dest/name
 if p.exists():continue
 text=(ROOT/'scripts'/old/name).read_text('utf8')
 if name!='network.py':text=text.replace(old,new).replace("sam2_g1_expansion_v2_20261005","sam2_g1_matched97_20261006")
 if name=='shared.py':
  text=text.replace("OUT = ROOT/'outputs/transformer_matched97_20261006'","import os\nVERSION = ROOT/'outputs/transformer_matched97_20261006'\nARM=os.environ.get('G1_BATCH_ARM','new')\nassert ARM in ['old','new']\nOUT = VERSION/ARM")
 p.write_text(text,encoding='utf8')
print(dest)
