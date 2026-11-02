"""Review new time ranges before annotation, independent of forecast scores."""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('screen_base',ROOT/'scripts/g1_street_candidates_20261001/screen.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=ROOT/'outputs/sam2_g1_expansion_20261005';OUT.mkdir(parents=True,exist_ok=True)
base.OUT=OUT
RANGES=[('150809',35,40),('150809',45,55),('151153',25,30),('151153',40,55),
        ('163303',5,10),('163303',50,55),('163303',90,95),('165112',60,65),
        ('194648',5,10),('194648',145,150),('201451',105,110),('205416',15,20),('205416',25,40),
        ('192702',191,201),('192702',215,229),('192702',235,245),('192702',169,181),
        ('193709',74,87),('164557',65,75),('193956',51,61)]


def main():
    rows=[];signals={}
    for sid,start,end in RANGES:
        if sid not in signals:signals[sid]=base.signals(sid)
        p=OUT/'screen_sheets'/f'{sid}_{start:03d}_{end:03d}.jpg'
        if not p.exists():base.sheet((sid,start,end))
        rows.append(dict(id=sid,start_s=start,end_s=end,metrics=base.metrics(signals[sid],start,end),sheet=str(p)))
        print(sid,start,end,flush=True)
    (OUT/'screen_candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
