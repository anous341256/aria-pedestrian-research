from pathlib import Path
from PIL import Image
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def main():
 dest=OUT/'review';dest.mkdir(exist_ok=True)
 rows=json.loads((OUT/'sample_locked.json').read_text('utf8'))
 for n in range(0,len(rows),5):
  batch=rows[n:n+5];paths=[OUT/'samples'/r['sample_id']/'context.jpg' for r in batch]
  if not all(p.exists() for p in paths):continue
  canvas=Image.new('RGB',(1600,280*len(batch)),(245,245,245))
  for j,p in enumerate(paths):canvas.paste(Image.open(p),(0,j*280))
  canvas.save(dest/f'context_{n//5+1:02d}.jpg',quality=94)
 print('context pages',len(list(dest.glob('context_*.jpg'))))
 tracks=[]
 for r in rows:
  p=OUT/'samples'/r['sample_id']/'proposals.json'
  if p.exists():tracks += [t for t in json.loads(p.read_text('utf8')) if t['automatic_eligible']]
 for n in range(0,len(tracks),6):
  batch=tracks[n:n+6];canvas=Image.new('RGB',(1470,230*len(batch)),(245,245,245))
  for j,t in enumerate(batch):canvas.paste(Image.open(OUT/'samples'/t['sample_id']/f'track_{t["track"]:03d}.jpg'),(0,j*230))
  canvas.save(dest/f'targets_{n//6+1:03d}.jpg',quality=94)
 (dest/'targets_manifest.json').write_text(json.dumps([dict(page=i//6+1,key=t['key']) for i,t in enumerate(tracks)],indent=2),'utf8')
 print('targets',len(tracks))
if __name__=='__main__':main()
