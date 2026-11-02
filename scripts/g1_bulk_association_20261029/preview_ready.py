from bulk_common import *
from render_bulk import cv2,color,text

def main():
 inv=read(OUT/'inventory.json');ready=[c for c in inv['clips'] if (OUT/'quality'/f'{c["clip"]}.json').exists()];panels=[]
 remaining='--remaining' in sys.argv
 if remaining:
  reviewed=set(read(OUT/'preview_ready/index.json')['clips']);ready=[c for c in ready if c['clip'] not in reviewed]
 for c in ready:
  cid=c['clip'];rows=read(OUT/'tracking'/f'{cid}.json')['rows'];entries=[e for e in inv['entries'] if e['clip']==cid];q=read(OUT/'quality'/f'{cid}.json');cap=cv2.VideoCapture(str(ROOT/c['input']/'rgb.mp4'));canvas=np.zeros((230,960,3),np.uint8)
  for j,f in enumerate([0,10,20]):
   cap.set(cv2.CAP_PROP_POS_FRAMES,c['lo']+f);ok,im=cap.read();assert ok;im=cv2.resize(im,(320,200));target={t['target_key']:t for t in entries[f]['targets']}
   for r in rows:
    if r['frame']!=f:continue
    t=target[r['uid']];b=(np.array(t['box'])*np.array([.125,200/1920,.125,200/1920])).astype(int);co=color(r['track_id']);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),co,1);text(im,f'T{r["track_id"]}',int(b[0]),max(11,int(b[1])-2),.36,co)
   canvas[30:230,j*320:(j+1)*320]=im;text(canvas,f'{cid} / frame {f} / windows {len(q["windows"])}',j*320+4,19,.43)
  cap.release();panels.append(canvas)
 out=OUT/('preview_remaining' if remaining else 'preview_ready');out.mkdir(exist_ok=True)
 for i in range(0,len(panels),6):cv2.imwrite(str(out/f'page_{i//6+1:02d}.jpg'),np.concatenate(panels[i:i+6]))
 write(out/'index.json',dict(clips=[c['clip'] for c in ready],pages=(len(panels)+5)//6));print('PREVIEW',len(ready))
if __name__=='__main__':main()
