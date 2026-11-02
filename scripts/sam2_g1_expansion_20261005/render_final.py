"""Full-clip reviewed mask videos and one 70-second compilation."""
from qa import *
from common import encoder,finish_encoder

def main():
 dest=OUT/'presentation';dest.mkdir(exist_ok=True)
 selected={f'{x["candidate"]:02d}':x for x in read_json(OUT/'selected_clips.json')}
 quality={(x['clip'],x['id'],x['frame']):x for x in read_json(OUT/'dataset/frame_quality.json')}
 total=read_json(OUT/'dataset/summary.json')['totals'];W,H=1280,1060
 allenc,alllog=encoder(dest/'ALL_10_CLIPS_70s.mp4',W,H,10,20);cards=[];media=[]
 for cid in CLIPS:
  info=selected[cid];folder=OUT/cid/'final';rows=v.load(folder/'masks_rle.jsonl.gz');census=read_json(folder/'census.json');ds=v.load(folder/'decisions.jsonl.gz');masks=collections.defaultdict(list);logs={(d['frame'],d['id']):d for d in ds};cols={e['id']:v.color(int(e['id'][1:])-1) for e in census['segments']}
  for r in rows:masks[r['frame']].append(r)
  cap=cv2.VideoCapture(str(OUT/cid/'input/rgb.mp4'));N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));enc,elog=encoder(dest/f'{cid}_near_people_reviewed.mp4',W,H,10,20)
  for f in range(N):
   ok,frame=cap.read();assert ok
   im,ids=v.overlay(frame,masks[f],cols,logs,f);canvas=np.full((H,W,3),(25,23,20),np.uint8);canvas[65:1025]=im
   clear=[i for i in ids if quality[cid,i,f]['clear']];partial=[i for i in ids if i not in clear]
   # Mark fragments explicitly; no discarded identity receives a fabricated mask.
   for r in masks[f]:
    if r['id'] in partial:
     x,y=int(r['box'][0]/2),max(15,int(r['box'][1]/2)-15)
     cv2.putText(canvas,'partial / review',(x,y+65),0,.4,(60,190,255),1,cv2.LINE_AA)
   title=info['visual_observation'].split('，')[0];scene='静止人物对照' if cid=='19' else 'G1 候选 / 近处人物'
   canvas=v.text(canvas,[(16,5,f'{cid}  {scene}  |  {title}'),(16,35,f'{f/10:.1f} / {N/10:.1f} 秒   当前 {len(ids)} 个 mask   身体清晰候选：'+(', '.join(clear) or '—')),(16,1029,'SAM2 + V6 · 离线核查标注；partial / review 不直接进入清晰连续轨迹子集')],21)
   enc.stdin.write(canvas.tobytes());allenc.stdin.write(canvas.tobytes())
   if f==N//2:
    cv2.imwrite(str(dest/f'{cid}_preview.jpg'),canvas);cards.append(cv2.resize(canvas,(640,530)))
  cap.release();finish_encoder(enc,elog);media.append(dict(clip=cid,frames=N,fps=10,seconds=N/10,path=f'{cid}_near_people_reviewed.mp4'));print(cid,'VIDEO',N,flush=True)
 finish_encoder(allenc,alllog)
 contact=np.zeros((5*530,1280,3),np.uint8)
 for j,im in enumerate(cards):contact[j//2*530:(j//2+1)*530,j%2*640:(j%2+1)*640]=im
 cv2.imwrite(str(dest/'CONTACT_SHEET.jpg'),contact);write_json(dest/'media.json',dict(compilation='ALL_10_CLIPS_70s.mp4',clips=media,totals=total))
if __name__=='__main__':main()
