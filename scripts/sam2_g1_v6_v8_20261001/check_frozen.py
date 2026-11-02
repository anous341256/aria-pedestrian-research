"""Re-run original focused tests with all test artifacts redirected to this trial."""
import sys,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
here=Path(__file__).parent
for ver,folder,test in [('v6','sam2_motion_tracker_v1_20261001','test_core.py'),('v8','sam2_reliability_prior_20261001','test_policy.py')]:
    src=ROOT/'scripts'/folder/test
    code="import sys;from pathlib import Path;sys.path.insert(0,"+repr(str(src.parent))+");p=Path("+repr(str(src))+");s=p.read_text();s=s.replace('checks={}', 'OUT=Path('+repr("+repr(str(OUT/ver/'tests'))+")+')\\nchecks={}',1);exec(compile(s,str(p),'exec'))"
    # v8's test obtains Path through its imported run module only indirectly; provide it here.
    subprocess.run([sys.executable,'-c',code],check=True)
hashes={}
for local,src in [('core.py','sam2_motion_tracker_v1_20261001/core.py'),('tracker_geometry.py','sam2_reliability_prior_20261001/tracker_geometry.py'),('reliability.py','sam2_reliability_prior_20261001/reliability.py')]:
    a=hashlib.sha256((here/local).read_bytes()).hexdigest();b=hashlib.sha256((ROOT/'scripts'/src).read_bytes()).hexdigest();assert a==b
    hashes[local]=a
(OUT/'unit_checks.json').write_text(json.dumps(dict(status='passed',frozen_policy_sha256=hashes),indent=2))
