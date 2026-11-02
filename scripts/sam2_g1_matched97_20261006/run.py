"""Reuse the exact previous near-person/V6 annotation functions on new inputs."""
from pathlib import Path
import importlib.util,argparse
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_matched97_20261006'
spec=importlib.util.spec_from_file_location('near_previous',ROOT/'scripts/sam2_near_people_20261002/run.py')
near=importlib.util.module_from_spec(spec);spec.loader.exec_module(near)
near.OUT=OUT;near.SRC=OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clips',nargs='+',default=[f'{i:02d}' for i in range(32,42)]);args=parser.parse_args()
    cfg=near.read_json(ROOT/'outputs/sam2_near_people_20261002/protocol.json')
    cfg.update(version='sam2_g1_matched97_20261006',base_run_sha256=near.digest(ROOT/'scripts/sam2_near_people_20261002/run.py'),
               scope='10 new intervals, 91 seconds; raw visual screen then rectified census review. No forecasting retraining.',
               classification='Target-relative G1 candidates, boundary and static-control flags retained; not all masks are strict G1.',
               camera_split_warning='Existing full-data camera training included newly added recordings. Later evaluation needs new recording-held-out camera folds.')
    near.write_json(OUT/'protocol.json',cfg)
    with near.torch.inference_mode(),near.torch.autocast('cuda',dtype=near.torch.bfloat16):
        tracker=near.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for cid in args.clips:
            if (OUT/cid/'complete.json').exists():continue
            entries=near.read_json(OUT/cid/'census.json')['segments'];cal=near.np.load(OUT/cid/'input/reference.npz')
            cap=near.cv2.VideoCapture(str(OUT/cid/'input/rgb.mp4'));frames=[]
            while True:
                ok,im=cap.read()
                if not ok:break
                frames.append(im)
            cap.release();archive={};rows=[];logs=[]
            for batch in sorted({e['batch'] for e in entries}):
                pending=[]
                for e in [x for x in entries if x['batch']==batch]:
                    rr,ll=near.run_segment(tracker,cid,e,frames,cal,archive);rows+=rr;logs+=ll;pending+=rr
                for r in pending:archive.setdefault(r['id'],{})[r['frame']]=r
            near.inc.dumpgz(OUT/cid/'masks_rle.jsonl.gz',rows);near.inc.dumpgz(OUT/cid/'decisions.jsonl.gz',logs)
            near.write_json(OUT/cid/'complete.json',dict(rows=len(rows),people=len(archive),segments=len(entries)))
            del frames;near.torch.cuda.empty_cache();print(cid,'COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
