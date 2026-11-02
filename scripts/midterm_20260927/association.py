"""Does rotation help select the SAME observed person from fixed detections?
No detector changes, no tracker tuning. Partial assistant-reviewed identities.
"""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,CONFIG,read_json,write_json,apply_h,iou

METHODS=['none','aria_rotation','rgb_homography','aria_rgb']
def center(b):b=np.asarray(b);return (b[:2]+b[2:4])/2

def main(strict=False):
    output=OUT/('results/association_unique' if strict else 'results/association');output.mkdir(exist_ok=True);rows=[];excluded=[]
    for spec in read_json(CONFIG)['clips']:
        clip=OUT/'clips'/spec['id'];ref=np.load(clip/'reference.npz');K=ref['K'];R=ref['R_world_camera'];geo=np.load(clip/'geometry/transforms.npz')
        detections=read_json(clip/'yolo/detections.json')['frames'];tracks=read_json(clip/'annotation_v1/tracks.json')['tracks']
        for t in tracks:
            rs={r['frame_id']:r for r in t['boxes']}
            for f,r in rs.items():
                if f-1 not in rs:continue
                ds=[d for d in detections[f] if d[4]>=.15]
                if len(ds)<2:continue
                overlaps=[iou(d[:4],r['bbox_xyxy']) for d in ds];true=int(np.argmax(overlaps))
                if overlaps[true]<.5:continue
                if strict:
                    ambiguous=[k for k,b in enumerate(ds) if k!=true and (iou(b[:4],ds[true][:4])>.2 or
                        (r['bbox_xyxy'][0]<=center(b)[0]<=r['bbox_xyxy'][2] and r['bbox_xyxy'][1]<=center(b)[1]<=r['bbox_xyxy'][3]))]
                    if ambiguous:
                        excluded.append({'clip':spec['id'],'track':t['track_id'],'frame':f,'reason':'overlapping/duplicate competing detections; identity distinction ambiguous','candidates':ambiguous});continue
                prev=center(rs[f-1]['bbox_xyxy']);Haria=K@R[f].T@R[f-1]@np.linalg.inv(K)
                Hs={'none':np.eye(3),'aria_rotation':Haria,'rgb_homography':geo['rgb_homography'][f],'aria_rgb':geo['aria_rgb'][f]}
                row={'clip':spec['id'],'recording':spec['recording'],'split':spec['split'],'track':t['track_id'],'frame':f,
                     'candidates':len(ds),'true_index':true,'true_iou_to_reference':overlaps[true],'common_rgb_valid':bool(geo['joint_valid'][f]),
                     'previous_box':rs[f-1]['bbox_xyxy'],'true_box':r['bbox_xyxy'],'detection_boxes':ds,'predicted_centers':{}}
                for m,H in Hs.items():
                    pred=apply_h([prev],H)[0];dist=np.linalg.norm(np.array([center(d) for d in ds])-pred,axis=1)
                    order=np.argsort(dist);rank=int(np.flatnonzero(order==true)[0])+1
                    row[m+'_top1']=bool(order[0]==true);row[m+'_rank']=rank;row[m+'_residual']=float(dist[true]);row[m+'_selected']=int(order[0]);row['predicted_centers'][m]=pred.tolist()
                rows.append(row)
    write_json(output/'pairs.json',rows);write_json(output/'excluded.json',excluded);summary=[]
    for spec in read_json(CONFIG)['clips']:
        subset=[r for r in rows if r['clip']==spec['id']];common=[r for r in subset if r['common_rgb_valid']]
        summary.append({'clip':spec['id'],'split':spec['split'],'all_available_pairs':len(subset),'common_pairs':len(common),
            'aria_vs_none_all':{m:{'top1':float(np.mean([r[m+'_top1'] for r in subset])),'median_residual_px':float(np.median([r[m+'_residual'] for r in subset]))} for m in ['none','aria_rotation']} if subset else {},
            'common_support':{m:{'top1':float(np.mean([r[m+'_top1'] for r in common])),'median_rank':float(np.median([r[m+'_rank'] for r in common])),'median_residual_px':float(np.median([r[m+'_residual'] for r in common]))} for m in METHODS} if common else {}})
    write_json(output/'summary.json',{'clips':summary,'strict_unique_target':strict,'excluded_ambiguous_pairs':len(excluded),
      'amendment':'After visual inspection revealed a duplicate detection on the same person was counted as wrong, this secondary analysis excludes ambiguous pairs for ALL methods: another detection IoU>0.2 with true detection OR its center inside reviewed target box. Original unrestricted results are preserved. Does not establish full tracking accuracy.' if strict else None,
      'protocol':'Closest predicted CENTER in native pixels, fixed >=.15 detections, at least 2 candidates. True candidate = highest IoU>=.5 to reviewed box. No learned appearance or future track state. Same candidate set. Offline MPS reference.',
      'limitations':'Assistant-prechecked partial labels are not independent gold; selected long surviving tracks and detector-derived boxes bias this diagnostic. Single-link top1 is not HOTA/IDF1/full MOT accuracy. Aria+RGB uses current RGB and offline MPS, NOT RGB-only deployment.'})
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');xx=np.arange(len(summary))
    for j,m in enumerate(METHODS):ax.bar(xx+(j-1.5)*.18,[s['common_support'].get(m,{}).get('top1',np.nan)*100 for s in summary],.18,label=m)
    ax.set_xticks(xx,[s['clip'] for s in summary],rotation=30,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Correct candidate top-1 (%)');ax.set_title('Fixed detections, SAME valid pair support | partial silver diagnostic');ax.legend();fig.savefig(output/'top1_comparison.png',dpi=170);plt.close(fig)
    # Predeclared qualitative selector: test-set successes/failures ranked by raw displacement.
    choices={
      'rotation_helps':sorted([r for r in rows if r['split']=='test' and not r['none_top1'] and r['aria_rotation_top1']],key=lambda r:-r['none_residual']),
      'rotation_hurts':sorted([r for r in rows if r['split']=='test' and r['none_top1'] and not r['aria_rotation_top1']],key=lambda r:-r['aria_rotation_residual'])}
    for tag,rr in choices.items():
        for j,r in enumerate(rr[:3]):
            cap=cv2.VideoCapture(str(OUT/'clips'/r['clip']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,im=cap.read();cap.release();assert ok;im=cv2.resize(im,(1280,960))
            for k,b in enumerate(r['detection_boxes']):
                pts=(np.asarray(b[:4])/2).astype(int);c=(40,255,70) if k==r['true_index'] else (130,130,130)
                cv2.rectangle(im,tuple(pts[:2]),tuple(pts[2:]),c,3 if k==r['true_index'] else 1)
            for m,col in [('none',(40,60,255)),('aria_rotation',(255,220,20))]:
                p=np.array(r['predicted_centers'][m])/2;q=center(r['detection_boxes'][r[m+'_selected']])/2
                cv2.drawMarker(im,tuple(p.astype(int)),col,cv2.MARKER_CROSS,20,3);cv2.arrowedLine(im,tuple(p.astype(int)),tuple(q.astype(int)),col,2,tipLength=.15)
            cv2.rectangle(im,(0,0),(1280,70),(0,0,0),-1)
            for k,line in enumerate([f'POST-HOC {tag}: {r["clip"]} P{r["track"]} f={r["frame"]}',
                 'Green=true candidate (partial silver) | Red=no compensation | Cyan=Aria rotation',
                 'Cross=predicted center; arrow=closest selected detection. Same candidates, no threshold tuning.']):
                cv2.putText(im,line,(10,21+k*22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            cv2.imwrite(str(output/f'{tag}_{j+1}.jpg'),im);write_json(output/f'{tag}_{j+1}.json',r)
    print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');a=p.parse_args();main(a.strict)
