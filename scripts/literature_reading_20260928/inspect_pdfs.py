"""Read-only paper inventory and local text extraction; preserves source PDFs."""
import hashlib, json
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'行人轨迹预测相关论文'
OUT=ROOT/'tmp/pdfs/reading_20260928'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(BASE.glob('*.pdf')):
    reader=PdfReader(p)
    first=reader.pages[0].extract_text() or ''
    row={'file':p.name,'pages':len(reader.pages),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'first_page_start':first[:1100]}
    rows.append(row)
    print(json.dumps({'file':p.name,'pages':len(reader.pages),'start':first[:420]},ensure_ascii=False))
    if p.name=='Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf':
        for n,page in enumerate(reader.pages,1):
            (OUT/f'neumann_page_{n:02d}.txt').write_text(page.extract_text(extraction_mode='layout'),encoding='utf-8')
(OUT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
