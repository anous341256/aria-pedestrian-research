from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def read(p):return json.loads(p.read_text('utf8'))
def main():
 review=read(OUT/'visual_review.json');dest=OUT/'review/qc';dest.mkdir(exist_ok=True)
 groups={'early':[],'later':[]}
 for sample in read(OUT/'sample_locked.json'):
  sid=sample['sample_id'];d=OUT/'samples'/sid
  if sid in review['global_exclusions'] or not (d/'segmentation.json').exists():continue
  seg=read(d/'segmentation.json')
  for t in read(d/'proposals.json'):
   if not t['automatic_eligible'] or t['key'] in review['target_exclusions']:continue
   groups['early' if int(sid[1:])<=39 else 'later'].append(t)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for group,tracks in groups.items():
  for n in range(0,len(tracks),6):
   batch=tracks[n:n+6];canvas=Image.new('RGB',(1840,230*len(batch)),(245,245,245));draw=ImageDraw.Draw(canvas)
   for j,t in enumerate(batch):
    d=OUT/'samples'/t['sample_id'];canvas.paste(Image.open(d/f'track_{t["track"]:03d}.jpg'),(0,j*230))
    for k,f in enumerate([0,5]):
     im=Image.open(d/f'mask_{t["track"]:03d}_{f:02d}.jpg');im.thumbnail((180,190));canvas.paste(im,(1470+185*k,j*230+35));draw.text((1470+185*k,j*230+3),f'SAM {f}',font=font,fill='black')
   canvas.save(dest/f'{group}_{n//6+1:03d}.jpg',quality=94)
  (dest/f'{group}_manifest.json').write_text(json.dumps([dict(page=i//6+1,key=t['key']) for i,t in enumerate(tracks)],indent=2),'utf8')
  print(group,len(tracks))
if __name__=='__main__':main()
