"""Exact frozen near-person/V6 propagation on the newly reviewed census."""
from pathlib import Path
import importlib.util,argparse,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_remaining_all_20261007'
spec=importlib.util.spec_from_file_location('near_previous',ROOT/'scripts/sam2_near_people_20261002/run.py')
near=importlib.util.module_from_spec(spec);spec.loader.exec_module(near);near.OUT=OUT;near.SRC=OUT
def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+');args=p.parse_args();clips=args.clips or [c['clip'] for c in near.read_json(OUT/'selected_clips.json')]
 cfg=near.read_json(ROOT/'outputs/sam2_near_people_20261002/protocol.json')
 cfg.update(version=OUT.name,scope='225 remaining camera-eligible time windows, 118 intervals, 311.8 seconds; no training.',annotation='Frozen V6 with current-frame previous-batch negatives and bidirectional reference geometry; provisional detector/census prompts, agent review only where explicitly recorded; user review pending.',base_run_sha256=near.digest(ROOT/'scripts/sam2_near_people_20261002/run.py'))
 near.write_json(OUT/'protocol.json',cfg)
 with near.torch.inference_mode(),near.torch.autocast('cuda',dtype=near.torch.bfloat16):
  tracker=near.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
  for cid in clips:
   if (OUT/cid/'complete.json').exists():continue
   entries=near.read_json(OUT/cid/'census.json')['segments'];cal=near.np.load(OUT/cid/'input/reference.npz');frames=[]
   if entries:
    cap=near.cv2.VideoCapture(str(OUT/cid/'input/rgb.mp4'))
    while True:
     ok,im=cap.read()
     if not ok:break
     frames.append(im)
    cap.release()
   archive={};rows=[];logs=[]
   for batch in sorted({e['batch'] for e in entries}):
    pending=[]
    for e in [x for x in entries if x['batch']==batch]:
     rr,ll=near.run_segment(tracker,cid,e,frames,cal,archive);rows+=rr;logs+=ll;pending+=rr
    for r in pending:archive.setdefault(r['id'],{})[r['frame']]=r
   near.inc.dumpgz(OUT/cid/'masks_rle.jsonl.gz',rows);near.inc.dumpgz(OUT/cid/'decisions.jsonl.gz',logs)
   near.write_json(OUT/cid/'complete.json',dict(rows=len(rows),people=len(archive),segments=len(entries),provisional=True));del frames;near.torch.cuda.empty_cache();print(cid,'COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
