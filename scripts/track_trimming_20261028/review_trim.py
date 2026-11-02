from trim_tracks import *
sys.path.insert(0,str(ROOT/'scripts/partial_appearance_20261028'))
from render_versions import zh,txt
def fit(im,w,h):
 s=min(w/im.shape[1],h/im.shape[0]);r=cv2.resize(im,(max(1,round(im.shape[1]*s)),max(1,round(im.shape[0]*s))));out=np.full((h,w,3),30,np.uint8);y=(h-r.shape[0])//2;x=(w-r.shape[1])//2;out[y:y+r.shape[0],x:x+r.shape[1]]=r;return out
def main():
 inv=read(BASE/'inventory.json');segments=read(TRIM/'retained_segments.json');by=collections.defaultdict(list)
 for s in segments:by[s['clip']].append(s)
 frames={};entries={(e['clip'],e['frame']):e for e in inv['entries']}
 for cid,ss in by.items():
  fs={f for s in ss for f in [s['start'],(s['start']+s['end'])//2,s['end']]};cap=cv2.VideoCapture(str(ROOT/entries[(cid,0)]['input']/'rgb.mp4'))
  for f in range(max(fs)+1):
   ok,im=cap.read();assert ok
   if f in fs:frames[cid,f]=im
  cap.release()
 pres=TRIM/'presentation'
 for page in range((len(segments)+4)//5):
  ss=segments[page*5:page*5+5];sheet=np.full((70+len(ss)*300,1440,3),20,np.uint8);sheet=zh(sheet,f'保留轨迹抽查 {page+1}：每段起点 / 中点 / 终点，右半为原 SAM mask',(15,12),27)
  for row,s in enumerate(ss):
   cid=s['clip'];fs=[s['start'],(s['start']+s['end'])//2,s['end']]
   for col,f in enumerate(fs):
    e=entries[cid,f];uid=s['uids'][f-s['start']];t=next(t for t in e['targets'] if t['target_key']==uid);x1,y1,x2,y2=map(int,t['box']);im=frames[cid,f];m=decode(t['rle']).astype(bool);masked=im.copy();masked[~m]=127;y=70+row*300;x=col*480;txt(sheet,f'{cid} T{s["original_track_id"]} f{f} [{s["start"]}-{s["end"]}]',(x+7,y+20),size=.53);sheet[y+30:y+290,x:x+240]=fit(im[y1:y2,x1:x2],240,260);sheet[y+30:y+290,x+240:x+480]=fit(masked[y1:y2,x1:x2],240,260)
  cv2.imwrite(str(pres/f'retained_review_{page+1:02d}.jpg'),sheet)
 print('REVIEW',len(segments),'segments',len(frames),'unique frames')
if __name__=='__main__':main()
