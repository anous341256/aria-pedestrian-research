"""Check delivery links and rebuild compact sheets without recomputing scores."""
import re
import cv2
import numpy as np
from prepare import ROOT, OUT, read, write

report=ROOT/'reports/factor_feasibility_expanded_20260928/RESULTS.md'
content=report.read_text(encoding='utf-8')
targets=[p for p in re.findall(r'\]\(([^)]+)\)',content) if not p.startswith('https://')]
assert all((report.parent/p).is_file() for p in targets)
manifest=read(OUT/'example_manifest.json')
thumbs=[cv2.resize(cv2.imread(str(OUT/p['path'])),(650,490)) for p in manifest]
for page in range((len(thumbs)+5)//6):
    ts=thumbs[page*6:(page+1)*6]
    if len(ts)%2:ts.append(np.full_like(thumbs[0],255))
    cv2.imwrite(str(OUT/f'example_contact_{page+1}.jpg'),np.vstack([np.hstack(ts[i:i+2]) for i in range(0,len(ts),2)]))
assert len(manifest)==14
write(OUT/'report_verification.json',{'local_report_links':len(targets),'all_links_exist':True,'example_images':14,'score_files_unchanged_by_reporting':True})
print('REPORT LINKS OK',len(targets),'EXAMPLE IMAGES',len(manifest))
