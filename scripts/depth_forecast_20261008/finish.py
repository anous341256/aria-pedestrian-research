"""Complete requested reports and fixed-case visualization after dependent jobs finish."""
import json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';HERE=Path(__file__).parent
while not ((OUT/'depth_all21.npz').exists() and (OUT/'forecast/verification.json').exists()):time.sleep(10)
with (OUT/'finish.log').open('w') as f:
    for name in ['audit_report.py','calibration_audit.py','render.py']:
        subprocess.run([sys.executable,str(HERE/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
(OUT/'delivery_complete.json').write_text(json.dumps(dict(complete=True,report='reports/depth_forecast_20261008/RESULTS.md',video='outputs/depth_forecast_20261008/presentation/depth_ablation_fixed6.mp4'),indent=2))
print('DELIVERY COMPLETE',flush=True)
