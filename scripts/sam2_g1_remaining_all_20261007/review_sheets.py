from pathlib import Path
import json,math
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_remaining_all_20261007'
def main():
 clips=json.loads((OUT/'selected_clips.json').read_text('utf8'));done=[c for c in clips if (OUT/c['clip']/'discovery_complete.json').exists()];dest=OUT/'review_sheets';dest.mkdir(exist_ok=True)
 for j in range(0,len(done),4):
  rows=done[j:j+4]
  if len(rows)<4 and len(done)!=len(clips):break
  path=dest/f'context_{j//4+1:03d}.jpg';canvas=Image.new('RGB',(1600,len(rows)*280),'white')
  for k,c in enumerate(rows):canvas.paste(Image.open(OUT/c['clip']/'context.jpg'),(0,k*280))
  canvas.save(path,quality=94)
 tracks=[p for c in done for p in sorted((OUT/c['clip']).glob('track_*.jpg'))]
 index=[]
 for j in range(0,len(tracks),8):
  rr=tracks[j:j+8]
  if len(rr)<8 and len(done)!=len(clips):break
  path=dest/f'tracks_{j//8+1:03d}.jpg';canvas=Image.new('RGB',(1400,len(rr)*210),'white')
  for k,p in enumerate(rr):canvas.paste(Image.open(p),(0,k*210))
  canvas.save(path,quality=94);index.append(dict(page=path.name,tracks=[f'{p.parent.name}:{int(p.stem.split("_")[1])}' for p in rr]))
 (dest/'index.json').write_text(json.dumps(dict(clips=[c['clip'] for c in done],track_pages=index),indent=2),'utf8')
 print('ready',len(done),'clips',len(tracks),'tracks',len(index),'track pages')
if __name__=='__main__':main()
