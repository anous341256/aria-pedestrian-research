"""One bounded re-anchor of a clear long trajectory; preserve first review."""
from run import near as n,OUT
import copy,shutil

def main():
 cid='13';folder=OUT/cid;dest=folder/'final'
 if (dest/'polish.json').exists():return
 backup=folder/'review_v1';shutil.copytree(dest,backup)
 census=n.read_json(dest/'census.json');entry=copy.deepcopy(next(e for e in census['segments'] if e['id']=='N01'))
 entry.update(start=43,end=54,anchor_frame=54,segment='N01_POLISH');entry['rows']=[r for r in entry['rows'] if 43<=r['frame']<=54];entry['anchor_box']=next(r['box'] for r in entry['rows'] if r['frame']==54)
 rows={(r['id'],r['frame']):r for r in n.inc.loadgz(dest/'masks_rle.jsonl.gz')};logs={(r['id'],r['frame']):r for r in n.inc.loadgz(dest/'decisions.jsonl.gz')}
 cap=n.cv2.VideoCapture(str(folder/'input/rgb.mp4'));frames=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  frames.append(im)
 cap.release();archive={}
 for r in rows.values():
  if r['id']!='N01':archive.setdefault(r['id'],{})[r['frame']]=r
 with n.torch.inference_mode(),n.torch.autocast('cuda',dtype=n.torch.bfloat16):
  tracker=n.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda');n.OUT=OUT/'polish_runs'
  rr,dd=n.run_segment(tracker,cid,entry,frames,n.np.load(folder/'input/reference.npz'),archive)
  chosen={r['frame']:r for r in rr};ds={r['frame']:r for r in dd};events=[]
  for f in range(43,51):
   r=chosen.get(f);d=ds.get(f,{});q=(r or {}).get('quality',{})
   if not r or 'UNCERTAIN' in d.get('status','') or q.get('guide_iou',0)<.6:continue
   r.update(segment='N01_S01',revision=2,donor_eligible=False);d.update(segment='N01_S01',revision=2,status='REVIEW_REANCHOR_'+d['status'])
   rows['N01',f]=r;logs['N01',f]=d;events.append(dict(frame=f,guide_iou=q['guide_iou']))
  n.inc.dumpgz(dest/'masks_rle.jsonl.gz',sorted(rows.values(),key=lambda r:(r['frame'],r['id'])));n.inc.dumpgz(dest/'decisions.jsonl.gz',sorted(logs.values(),key=lambda r:(r['frame'],r['id'])))
  complete=n.read_json(dest/'complete.json');complete['rows']=len(rows);n.write_json(dest/'complete.json',complete)
  n.write_json(dest/'polish.json',dict(anchor=54,range=[43,54],replaced=events,previous_revision='review_v1',source='Full clear frame re-anchor; no interpolation or subtraction'))
  print(events,flush=True)
if __name__=='__main__':main()
