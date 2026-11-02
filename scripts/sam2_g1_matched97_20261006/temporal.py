"""Proposal temporal strips for explicit identity review before SAM propagation."""
from census import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',required=True);args=p.parse_args()
 for cid in args.clips:
  dest=OUT/cid/'temporal';dest.mkdir(exist_ok=True)
  props=read_json(OUT/cid/'near_proposals.json');cap=cv2.VideoCapture(str(OUT/cid/'input/rgb.mp4'));frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(im)
  cap.release();props=[t for t in props if len(t['rows'])>=8]
  for page in range(math.ceil(len(props)/8)):
   canvas=Image.new('RGB',(1400,8*180),(242,242,242));draw=ImageDraw.Draw(canvas)
   for j,t in enumerate(props[page*8:(page+1)*8]):
    y=j*180;draw.text((2,y+3),f"P{t['proposal_id']}\nf{t['first']}-{t['last']}",font=FONT,fill='black')
    rows=t['rows']
    for k,q in enumerate(np.unique(np.linspace(0,len(rows)-1,7).astype(int))):
     r=rows[q];x1,y1,x2,y2=map(int,r['box']);crop=Image.fromarray(cv2.cvtColor(frames[r['frame']][max(0,y1-10):min(1920,y2+10),max(0,x1-10):min(2560,x2+10)],cv2.COLOR_BGR2RGB));crop.thumbnail((178,145));x=130+k*180;canvas.paste(crop,(x,y));draw.text((x,y+150),str(r['frame']),font=FONT,fill='black')
   canvas.save(dest/f'page_{page+1}.jpg',quality=92)
  print(cid,len(props),'temporal strips',flush=True)
if __name__=='__main__':main()
