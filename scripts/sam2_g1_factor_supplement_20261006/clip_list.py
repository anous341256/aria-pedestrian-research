import argparse,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'outputs/sam2_g1_factor_supplement_20261006'
def chosen():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',type=int)
 args=p.parse_args();return args.clips or [x['candidate'] for x in json.loads((OUT/'selected_clips.json').read_text('utf8'))]
