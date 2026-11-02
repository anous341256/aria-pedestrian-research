"""Second expansion: inspect unused source intervals, without model-score selection."""
from pathlib import Path
import importlib.util,json
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_g1_expansion_v2_20261005';OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/g1_street_candidates_20261001/screen.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.OUT=OUT
RANGES=[('192702',215,229),('192702',235,250),('193709',62,74),('155313',178,193),
 ('164557',35,50),('164557',50,65),('151153',15,25),('151153',30,40),
 ('163303',8,20),('194648',150,162),('194648',215,227),('205416',20,30),
 ('193311',42,54),('201129',10,22),('201129',25,38),('193956',51,63),
 ('165112',20,32),('162538',85,97),('155313',220,232),('193956',30,42)]
def main():
 with ThreadPoolExecutor(max_workers=3) as pool:
  paths=list(pool.map(base.sheet,RANGES))
 cache={};rows=[]
 for (sid,lo,hi),p in zip(RANGES,paths):
  if sid not in cache:cache[sid]=base.signals(sid)
  rows.append(dict(id=sid,start_s=lo,end_s=hi,sheet=p,metrics=base.metrics(cache[sid],lo,hi)))
 (OUT/'screen_candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 print('Screened',len(rows),'new candidate intervals',flush=True)
if __name__=='__main__':main()
