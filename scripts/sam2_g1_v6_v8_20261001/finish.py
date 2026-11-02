"""Verify delivered media and add a comparison record without renumbering methods."""
from pathlib import Path
import sys,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001';P=OUT/'presentation'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
checks=[]
for k in range(0,11):
    name='all_10_comparison.mp4' if k==0 else f'{k:02d}_comparison.mp4'
    cap=cv2.VideoCapture(str(P/name));count=0;expected=530 if k==0 else read_json(OUT/f'{k:02d}'/'input/metadata.json')['frames']
    fps=cap.get(cv2.CAP_PROP_FPS);assert abs(fps-10)<1e-5
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(880,1920)
        if k==10 and count==37:cv2.imwrite(str(P/'case_10_037.jpg'),im)
        count+=1
    cap.release();assert count==expected
    checks.append(dict(file=name,frames=count,fps=fps,width=1920,height=880,sha256=digest(P/name)))
write_json(OUT/'media_verification.json',dict(status='passed',videos=checks))
registry=ROOT/'reports/sam2_tracking_history/registry.json';data=read_json(registry)
data['latest_comparison']='g1_v6_v8_20261001'
record=dict(id='g1_v6_v8_20261001',name='Frozen versions 6 and 8 on ten selected G1 clips',versions=[6,8],
    report='reports/sam2_g1_v6_v8_20261001/RESULTS.md',output_directory='outputs/sam2_g1_v6_v8_20261001',clips=10,initialized_tracks=29,
    result='No overall advantage established for version 8. V6 more continuous; V8 suppresses some bad masks but rejects visible people and still has ID transfer. Both fail post-exit re-identification in clip 10.',
    evaluation_status='Full outputs and logs; 50-time-point visual review plus selected failure inspection; no exhaustive GT; nonempty counts are not accuracy.')
comparisons=data.setdefault('comparisons',[])
if not any(c['id']==record['id'] for c in comparisons):comparisons.append(record)
write_json(registry,data)
print('All 11 delivered videos fully decoded and verified; comparison added to history.')
