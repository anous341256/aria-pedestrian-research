"""Rectify only the random sampled windows; never select on RGB content."""
from pathlib import Path
import sys,json,csv,time
import numpy as np,pandas as pd,cv2
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider,calibration
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def local(p):return Path('/mnt/'+p[0].lower()+p[2:].replace('\\','/')) if ':' in p else Path(p)
def main():
 rows=read(OUT/'sample_locked.json');cache={};begin=time.monotonic()
 for row in rows:
  dest=OUT/'samples'/row['sample_id'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'ready.json').exists():continue
  sid=row['id']
  if sid not in cache:
   provider=data_provider.create_vrs_data_provider(str(local(row['vrs'])));calib=provider.get_device_calibration().get_camera_calib('camera-rgb')
   target=calibration.get_linear_camera_calibration(2560,1920,float(calib.get_focal_lengths()[0]),'pinhole',calib.get_transform_device_camera())
   p=target.projection_params();K=np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]])
   mapping=list(csv.DictReader(open(local(row['timestamp_map']),encoding='utf-8-sig')));times=np.array([int(x['vrs_device_time_ns']) for x in mapping],np.int64)
   df=pd.read_csv(local(row['mps']));ts=np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64)
   cache[sid]=(calib,target,K,times,df,ts)
  calib,target,K,times,df,ts=cache[sid];ex=calib.get_transform_device_camera().to_matrix()
  fs=np.arange(row['start_frame'],row['start_frame']+21);ns=times[fs]
  ix=np.searchsorted(ts,ns).clip(1,len(ts)-1);ix=np.where(abs(ts[ix]-ns)<abs(ts[ix-1]-ns),ix,ix-1)
  rd=Rotation.from_quat(df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy()[ix]).as_matrix()
  R=rd@ex[:3,:3];pos=df[['tx_world_device','ty_world_device','tz_world_device']].to_numpy()[ix]+np.einsum('nij,j->ni',rd,ex[:3,3])
  np.savez_compressed(dest/'reference.npz',R=R,K=K,times_ns=ns,position=pos,T_device_camera=ex)
  cap=cv2.VideoCapture(str(local(row['video'])));cap.set(cv2.CAP_PROP_POS_FRAMES,int(fs[0]))
  for f in range(21):
   ok,bgr=cap.read();assert ok,(row['sample_id'],f)
   rgb=calibration.distort_by_calibration(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),target,calib)
   assert cv2.imwrite(str(dest/f'f{f:02d}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
  cap.release();write(dest/'ready.json',dict(sample=row,frames=21,seconds=time.monotonic()-begin))
  print(row['sample_id'],sid,row['start_s'],'elapsed',round(time.monotonic()-begin,1),flush=True)
if __name__=='__main__':main()
