from pathlib import Path
import json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
from audit import ROOT, OUT, read, save

SPEC=[('164557',180,185),('201451',40,50),('201451',55,60),
      ('151153',120,125),('150809',35,40),('150809',110,115),
      ('165112',90,100),('165112',155,165),('170022',90,100),('170022',120,130),
      ('150809',80,90),('192702',165,175),('164557',110,120),
      ('151153',20,30),('162538',10,20),('193709',35,45),('201129',15,25),('155313',75,85)]

def render(spec):
    sid,start,end=spec
    rec=next(r for r in read(OUT/'motion_inventory.json') if r['id']==sid)
    cap=cv2.VideoCapture(rec['video']); fps=cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps-10)<.001
    key=f'{sid}_{start:03d}_{end:03d}'
    # Dense frame sequence: 12 or 20 uniformly spaced observations including the final frame.
    ids=np.unique(np.linspace(round(start*fps),round(end*fps)-1,12 if end-start<=5 else 20).astype(int))
    sheet=Image.new('RGB',(1600,330*((len(ids)+3)//4)),(20,25,33)); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for j,f in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f)); ok,frame=cap.read(); assert ok,(sid,f)
        image=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(400,300)),cv2.COLOR_BGR2RGB))
        x=(j%4)*400; y=(j//4)*330
        sheet.paste(image,(x,y+30)); draw.text((x+6,y+4),f'{sid}  {f/fps:.1f}s',font=font,fill='white')
    cap.release()
    path=OUT/'segment_sheets'/f'{key}.jpg'; path.parent.mkdir(exist_ok=True); sheet.save(path,quality=90)
    return {'key':key,'recording':sid,'start_s':start,'end_s':end,'video':rec['video'],'sheet':str(path),'sample_frames':ids.tolist(),'windows':[w for w in rec['windows'] if w['start_s']<end and w['end_s']>start]}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(render,SPEC))
    save(OUT/'segment_evidence.json',results)
    print('Rendered',len(results),'segment sheets')
