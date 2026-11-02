"""Read-only extraction status; avoids Windows file-share locks."""
from pathlib import Path
import json,time
p=Path(__file__).resolve().parents[2]/'outputs/depth_forecast_20261008'
progress=json.loads((p/'extraction_progress.json').read_text()) if (p/'extraction_progress.json').exists() else {}
print(json.dumps(progress))
if (p/'extraction_complete.json').exists():print('COMPLETE', (p/'extraction_complete.json').read_text())
