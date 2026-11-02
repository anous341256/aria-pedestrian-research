from partial_common import *
from render_versions import zh,txt
import cv2
def crop_canvas(im,t):
 x1,y1,x2,y2=map(int,t['box']);m=decode(t['rle'])[y1:y2,x1:x2];c=im[y1:y2,x1:x2].copy();c[m==0]=127;h,w=c.shape[:2];s=224/max(h,w);ww,hh=round(w*s),round(h*s);out=np.full((224,224,3),127,np.uint8);out[(224-hh)//2:(224-hh)//2+hh,(224-ww)//2:(224-ww)//2+ww]=cv2.resize(c,(ww,hh));return out
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');oldcases=read(ROOT/'outputs/geometry_association_failures_20261027/selected_cases.json');entries={(e['clip'],e['frame']):e for e in inv['entries']};summary=[];pres=OUT/'presentation';pres.mkdir(exist_ok=True);comp=read(OUT/'comparison.json');changes=[];usage=[]
 for cl in inv['clips']:
  cid=cl['clip'];data=read(OUT/'tracking'/f'{cid}.json');rows=data['rows'];b={(r['source_id'],r['frame']) for r in comp['baseline']['per_clip'][cid]['changes']};n={(r['source_id'],r['frame']) for r in comp['partial']['per_clip'][cid]['changes']};changes.append(dict(clip=cid,removed_original_change_events=sorted(b-n),new_or_shifted_change_events=sorted(n-b)));usage.append(dict(clip=cid,matched=sum(r['action']=='matched' for r in rows),local_used=sum(r.get('local_weight',0)>0 for r in rows),total_pairs=len(data['pairs']),local_pair_support=sum(r.get('local_weight',0)>0 for r in data['pairs'])))
 for c in oldcases:
  cid=c['clip'];f=c['frame'];uid=c['catalog_event']['uid'];sid=c['source_id'];data=read(OUT/'tracking'/f'{cid}.json');rows=data['rows'];r=next(r for r in rows if r['uid']==uid);prev=[r0 for r0 in rows if r0['frame']<f and ref[r0['uid']]['source_id']==sid];previous=prev[-1] if prev else None;follow=[dict(frame=r0['frame'],track_id=r0['track_id'],action=r0['action']) for r0 in rows if abs(r0['frame']-f)<=4 and ref[r0['uid']]['source_id']==sid];record=dict(code=c['code'],clip=cid,frame=f,source_id=sid,new_row=r,previous_same_source=previous,local_timeline=follow);summary.append(record)
  # Visualize exact historical patch correspondences used in this actual decision.
  s=r.get('local_support')
  if not s:continue
  gu=s['gallery_uid'];gf=ref[gu]['frame'];assert gf<f;cap=cv2.VideoCapture(str(ROOT/entries[cid,0]['input']/'rgb.mp4'));ims={}
  for j in range(f+1):
   ok,im=cap.read();assert ok
   if j in [gf,f]:ims[j]=im
  cap.release();ta=next(t for t in entries[cid,gf]['targets'] if t['target_key']==gu);tb=next(t for t in entries[cid,f]['targets'] if t['target_key']==uid);a=crop_canvas(ims[gf],ta);bim=crop_canvas(ims[f],tb);feat=np.load(OUT/'patches'/f'{cid}.npz');loc={u:i for i,u in enumerate(feat['uids'])};xa=feat['xy'][loc[gu]];xb=feat['xy'][loc[uid]];canvas=np.full((760,1200,3),22,np.uint8);canvas=zh(canvas,f'{c["code"]} / 片段{cid}：实际参与匹配的历史局部与当前可见局部',(20,15),26);canvas=zh(canvas,f'历史帧{gf}（原标签{ref[gu]["source_id"]}）                  当前帧{f}（原标签{sid}）',(80,70),23);canvas[120:568,60:508]=cv2.resize(a,(448,448));canvas[120:568,692:1140]=cv2.resize(bim,(448,448))
  for ia,ib in s['pairs']:
   pa=tuple((xa[ia]*448+[60,120]).astype(int));pb=tuple((xb[ib]*448+[692,120]).astype(int));cv2.circle(canvas,pa,4,(0,235,255),-1);cv2.circle(canvas,pb,4,(0,235,255),-1);cv2.line(canvas,pa,pb,(70,160,70),1,cv2.LINE_AA)
  canvas=zh(canvas,f'整体 {r["global_appearance"]:.3f} → 融合 {r["appearance"]:.3f}；局部 {r["local_appearance"]:.3f}；对应点 {s["inliers"]}',(25,610),26);canvas=zh(canvas,'线表示通过空间一致性检查的局部对应，不是人体部位标注，也不保证身份正确。',(25,660),22);cv2.imwrite(str(pres/f'{c["code"]}_local_correspondence.jpg'),canvas)
 write(OUT/'original_failure_cases.json',summary);write(OUT/'changed_events.json',changes);write(OUT/'local_usage.json',usage)
 print('CASES',[(r['code'],r['new_row']['track_id'],None if r['previous_same_source'] is None else r['previous_same_source']['track_id'],r['new_row'].get('global_appearance'),r['new_row'].get('appearance'),r['new_row'].get('local_weight')) for r in summary]);print('USAGE',sum(r['local_used'] for r in usage))
if __name__=='__main__':main()
