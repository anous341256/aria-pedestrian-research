from screen import *

SPEC=[('192702',184,194),('192702',207,215),('192702',146,152),('192702',158,162),('192702',229,235),('193311',27,32),('164557',30,35),('164535',10,15),('193709',57,62),('155313',173,181)]
def render(item):
    n,(sid,start,end)=item
    cap=cv2.VideoCapture(INV[sid]['video']);frames=list(range(start*10,end*10,2))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);paths=[]
    for p in range(0,len(frames),24):
        subset=frames[p:p+24];im=Image.new('RGB',(1600,330*((len(subset)+3)//4)),(19,24,30));d=ImageDraw.Draw(im)
        for k,f in enumerate(subset):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
            x=k%4*400;y=k//4*330
            im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(bgr,(400,300)),cv2.COLOR_BGR2RGB)),(x,y+30))
            d.text((x+5,y+4),f'{n:02d} | {sid} | {f/10:.1f}s',font=font,fill='white')
        path=OUT/'dense_review'/f'{n:02d}_{sid}_{start}_{end}_p{p//24+1}.jpg';path.parent.mkdir(exist_ok=True);im.save(path,quality=91);paths.append(str(path))
    cap.release()
    a=np.load(OUT/f'pose_{sid}.npz');sig=tuple(a[k] for k in ['rel','stamp','q','valid','pos'])
    return {'candidate':n,'id':sid,'start_s':start,'end_s':end,'duration_s':end-start,'dense_review_sheets':paths,'sample_frames':frames,'metrics':metrics(sig,start,end)}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(render,enumerate(SPEC,1)))
    save(OUT/'refined_candidates.json',rows)
    for row in rows:print({k:v for k,v in row.items() if k not in ['sample_frames','dense_review_sheets']},flush=True)
