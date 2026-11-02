"""Fixed first window per recording, last historical frame; no metric selection."""
from depth_common import *
import cv2
def main():
 meta=read(BASE/'samples.json');plan=read(OUT/'inventory.json');entries={e['key']:e for e in plan['entries']};panels=[];selected=[]
 for rec in sorted({m['recording'] for m in meta}):
  i=next(i for i,m in enumerate(meta) if m['recording']==rec);m=meta[i];s=plan['sample_refs'][i];key=s['frame_keys'][15];tk=s['target_keys'][15];e=entries[key];done=read(OUT/'frames'/key/'complete.json');a=dict(np.load(ROOT/done['raw_path']));measurement=next(r for r in read(OUT/'frames'/key/'measurements.json') if r['target_key']==tk);r=next(t['rle'] for t in e['targets'] if t['target_key']==tk)
  mask=np.zeros((1920,2560),np.uint8);x0,y0,x1,y1=map(int,r[-4:]);mask[y0:y1+1,x0:x1+1]=np.repeat(np.arange(len(r[:-4]))%2,np.array(r[:-4],int)).reshape(y1-y0+1,x1-x0+1)
  cap=cv2.VideoCapture(str(ROOT/e['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][15]);ok,img=cap.read();assert ok;cap.release();image=cv2.resize(img,(640,480));overlay=image.copy();contours,_=cv2.findContours(cv2.resize(mask,(640,480),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,(80,220,80),2)
  disparity=a['disparity_px'];yy,xx=np.indices(disparity.shape);xr=xx-disparity;xi=np.clip(np.nan_to_num(xr),0,511).astype(int);support=a['support_left']&a['support_right'][yy,xi]&(xr>=0)&(xr<=511);z=(float(a['baseline_m'])*float(a['intrinsics'][0])/(disparity+1e-6)).astype(np.float32);valid=support&a['lr_consistent']&np.isfinite(z)&(z>0)&(disparity>0);y,x=np.nonzero(valid);v=z[y,x];fx,fy,cx,cy=a['intrinsics'];p=np.column_stack(((x-cx)*v/fx,(y-cy)*v/fy,v))@a['T_rgb_cv'][:3,:3].T+a['T_rgb_cv'][:3,3];h=p@a['K_rgb'].T;uv=h[:,:2]/h[:,2,None];inside=(p[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5);uv=np.rint(uv[inside]).astype(int);zz=p[inside,2];member=mask[uv[:,1],uv[:,0]]>0;uv=uv[member];zz=zz[member];color=cv2.applyColorMap(np.uint8((1-np.clip(np.log(np.maximum(zz,1))/np.log(15),0,1))*255),cv2.COLORMAP_TURBO).reshape(-1,3)
  order=np.argsort(zz)[::-1]
  for k in order:cv2.circle(overlay,tuple(np.rint(uv[k]*.25).astype(int)),1,tuple(map(int,color[k])),-1)
  cv2.drawContours(overlay,contours,-1,(80,220,80),1);panel=np.zeros((560,1280,3),np.uint8);panel[55:535,:640]=image;panel[55:535,640:]=overlay
  def txt(text,y):cv2.putText(panel,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2,cv2.LINE_AA)
  txt(f'{m["key"]} | {rec} | Last observed frame',25);txt('Left: SAM mask. Right: LR-valid depth samples (warm=near, cool=far).',50)
  status='valid' if measurement['usable'] else 'missing: '+','.join(measurement['reasons']);txt(f'{status} | median Z={measurement["z_median_m"]} | points={measurement["points"]}',555)
  panels.append(cv2.resize(panel,(960,420)));selected.append(dict(key=m['key'],frame=m['frames'][15],measurement=measurement))
 dest=OUT/'presentation';dest.mkdir(exist_ok=True);cv2.imwrite(str(dest/'fixed_6_depth_review.jpg'),np.vstack([np.hstack(panels[j:j+2]) for j in range(0,6,2)]));write(dest/'depth_review_selection.json',dict(selection='First existing sample of each recording; last historical frame. Fixed independently of results.',cases=selected))
if __name__=='__main__':main()
