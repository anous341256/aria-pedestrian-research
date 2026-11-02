"""Find unused same-recording intervals; no forecast errors used."""
from pathlib import Path
import importlib.util,json,cv2
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_matched97_20261006';OUT.mkdir(exist_ok=True,parents=True)
spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/g1_street_candidates_20261001/screen.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.OUT=OUT
RECORDS=['164557','151153','194648','163303','193709','192702','155313','201129','165112']

def main():
 used=[]
 for name in ['g1_street_candidates_20261001','sam2_g1_expansion_20261005','sam2_g1_expansion_v2_20261005']:
  used+=json.loads((ROOT/'outputs'/name/'selected_clips.json').read_text(encoding='utf8'))
 # Also avoid the earlier night tracking trial's already annotated 10-34 seconds.
 used.append(dict(id='201129',start_s=10,end_s=34))
 candidates=[];allrows=[]
 for sid in RECORDS:
  cap=cv2.VideoCapture(base.INV[sid]['video']);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/10;cap.release();sig=base.signals(sid);rows=[]
  for start in range(1,int(duration)-8):
   end=start+8
   if any(x['id']==sid and max(start-1,x['start_s'])<min(end+1,x['end_s']) for x in used):continue
   m=base.metrics(sig,start,end)
   if not m or m['pose_coverage']<.95 or m['max_1s_rotation_deg'] is None:continue
   row=dict(id=sid,start_s=start,end_s=end,metrics=m,rank=m['max_1s_rotation_deg']+.2*m['omega_p95_deg_s']);rows.append(row)
  keep=[]
  for r in sorted(rows,key=lambda x:x['rank']):
   if any(max(r['start_s'],x['start_s'])<min(r['end_s'],x['end_s']) for x in keep):continue
   keep.append(r)
   if len(keep)==4:break
  candidates+=keep;allrows+=rows
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row,path in zip(candidates,pool.map(base.sheet,[(x['id'],x['start_s'],x['end_s']) for x in candidates])):row['sheet']=path
 for name,obj in [('screen_candidates.json',candidates),('all_unused_candidates.json',allrows),('excluded_intervals.json',used)]:
  (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
 for x in candidates:print(x['id'],x['start_s'],x['end_s'],round(x['metrics']['max_1s_rotation_deg'],1),round(x['metrics']['omega_p95_deg_s'],1),flush=True)
if __name__=='__main__':main()
