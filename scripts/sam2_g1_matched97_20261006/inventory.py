"""Proposal tracklets and review sheets. These are discovery aids, not identities."""
from pathlib import Path
import sys,json,time
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'outputs/sam2_g1_matched97_20261006';OUT=ROOT/'outputs/sam2_g1_matched97_20261006'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json
def iou(a,b):
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def warp(b,H):
    a=np.array([[b[0],b[1],1],[b[2],b[3],1]]).T;q=H@a;q=q[:2]/q[2:]
    return np.r_[q.min(axis=1),q.max(axis=1)]
def main():
    from clip_list import chosen
    for c in chosen():
        cid=f'{c:02d}';dest=OUT/cid
        if (dest/'proposal_tracklets.json').exists():continue
        while not (dest/'detections.json').exists():time.sleep(2)
        ds=read_json(dest/'detections.json');cal=np.load(SRC/cid/'input/reference.npz');R,K=cal['R'],cal['K'];ki=np.linalg.inv(K)
        tracks=[]
        for row in ds:
            f=row['frame'];det=[d for d in row['detections'] if d['confidence']>=.18]
            active=[t for t in tracks if f-t['rows'][-1]['frame']<=4];cost=np.ones((len(active),len(det)))
            for i,t in enumerate(active):
                last=t['rows'][-1];b=warp(last['box'],K@R[f].T@R[last['frame']]@ki)
                for j,d in enumerate(det):cost[i,j]=1-iou(b,d['box'])
            assigned=set()
            if cost.size:
                for i,j in zip(*linear_sum_assignment(cost)):
                    if cost[i,j]<.82:active[i]['rows'].append(dict(frame=f,**det[j]));assigned.add(j)
            for j,d in enumerate(det):
                if j not in assigned:tracks.append(dict(proposal_id=len(tracks)+1,rows=[dict(frame=f,**d)]))
        for t in tracks:
            rows=t['rows'];strong=[r for r in rows if r['confidence']>=.35 and r['box'][3]-r['box'][1]>=24]
            t['eligible']=len(rows)>=3 and bool(strong)
            t['seed']=strong[0] if strong else max(rows,key=lambda r:r['confidence'])
            t['max_conf']=max(r['confidence'] for r in rows)
        write_json(dest/'proposal_tracklets.json',tracks)
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames={}
        for f in range(len(ds)):
            ok,im=cap.read();assert ok;frames[f]=im
        cap.release()
        eligible=[t for t in tracks if t['eligible']];cells=[]
        for t in eligible:
            s=t['seed'];x1,y1,x2,y2=map(int,s['box']);f=s['frame'];pad=max(12,(y2-y1)//5)
            crop=frames[f][max(0,y1-pad):min(1920,y2+pad),max(0,x1-pad):min(2560,x2+pad)]
            h,w=crop.shape[:2];scale=min(210/w,230/h);small=cv2.resize(crop,(max(1,int(w*scale)),max(1,int(h*scale))))
            tile=np.zeros((280,220,3),np.uint8);tile[38:38+small.shape[0],:small.shape[1]]=small
            cv2.putText(tile,f"T{t['proposal_id']} f{f}-{t['rows'][-1]['frame']}",(3,16),0,.45,(255,255,255),1)
            cv2.putText(tile,f"n={len(t['rows'])} p={t['max_conf']:.2f}",(3,32),0,.42,(255,255,255),1);cells.append(tile)
        for page in range((len(cells)+29)//30):
            sheet=np.zeros((5*280,6*220,3),np.uint8)
            for j,cell in enumerate(cells[page*30:(page+1)*30]):sheet[(j//6)*280:(j//6+1)*280,(j%6)*220:(j%6+1)*220]=cell
            cv2.imwrite(str(dest/f'proposal_review_{page+1}.jpg'),sheet)
        for f in [0,len(ds)//2,len(ds)-1]:
            im=frames[f].copy()
            for t in eligible:
                r=next((r for r in t['rows'] if r['frame']==f),None)
                if r:
                    x1,y1,x2,y2=map(int,r['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,255),2);cv2.putText(im,f"T{t['proposal_id']}",(x1,y1-4),0,.6,(0,255,255),2)
            cv2.imwrite(str(dest/f'discovery_{f:03d}.jpg'),cv2.resize(im,(1600,1200)))
        print(cid,'eligible',len(eligible),'all',len(tracks),flush=True)
if __name__=='__main__':main()
