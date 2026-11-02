"""Prepare immutable, blinded single-step association review inputs on E: only."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from core import METHODS, box_scores, warp_box

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/tracktrack_reproduction/aria_164557_first60s_v1'
ROT=ROOT/'outputs/aria_rotation_background/rectified_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def prepare(output, random_n=60, hard_n=30):
    if output.exists():raise FileExistsError('Use a new directory; no overwrites')
    with np.load(ROT/'relative_rotations.npz') as f:
        rotations=f['R_current_from_previous']; K=f['K_rect']
    meta=json.loads((BASE/'input/manifest.json').read_text(encoding='utf-8'))
    assert np.allclose(K,meta['geometry']['target']['K'])
    cmc=np.loadtxt(BASE/'cmc/GMC-Aria.txt',delimiter='\t',usecols=range(7))
    assert len(cmc)==len(rotations)==600 and np.array_equal(cmc[:,0],np.arange(600))
    assert np.allclose(cmc[0,1:],[1,0,0,0,1,0])
    with (BASE/'input/frame_timestamps.csv').open(encoding='utf-8-sig',newline='') as f:times=list(csv.DictReader(f))
    with (BASE/'cmc/quality.tsv').open(encoding='utf-8-sig',newline='') as f:quality=list(csv.DictReader(f,delimiter='\t'))
    assert len(times)==len(quality)==600
    caches=[]; digest=hashlib.sha256()
    for frame in range(600):
        path=BASE/f'tracking/detections/{frame+1:06d}.npz'
        digest.update(f'{frame+1}:{sha(path)}\n'.encode())
        with np.load(path,allow_pickle=False) as f:
            assert int(f['frame_id'])==frame+1
            det=f['tracker_input80'][:,:6].astype(float)
            assert np.isfinite(det).all() and np.all(det[:,2:4]>det[:,:2])
            caches.append(det)
    pool=[]; excluded=Counter()
    for frame in range(1,600):
        if not np.isfinite(rotations[frame]).all():
            excluded['invalid_pose_frame_pairs']+=1;continue
        if not len(caches[frame]):
            excluded['no_current_candidates_frame_pairs']+=1;continue
        H=K@rotations[frame]@np.linalg.inv(K)
        rgb=np.vstack([cmc[frame,1:].reshape(2,3),[0,0,1]])
        angle=float(np.rad2deg(np.arccos(np.clip((np.trace(rotations[frame])-1)/2,-1,1))))
        seconds=(int(times[frame]['device_timestamp_ns'])-int(times[0]['device_timestamp_ns']))/1e9
        for source_index,det in enumerate(caches[frame-1]):
            source=det[:4]; methods={}
            for method,M in zip(METHODS,[np.eye(3),rgb,H]):
                predicted=warp_box(source,M)
                if predicted is None:break
                methods[method]=box_scores(source,predicted,caches[frame][:,:4])
            if len(methods)!=3:
                excluded['nonprojectable_source_boxes']+=1;continue
            q={'id':f'f{frame:04d}_s{source_index:03d}', 'frame':frame,'previous_frame':frame-1,
               'source_index':source_index,'source_box':source.tolist(),'source_score':float(det[4]),
               'candidates':[{'index':i,'box':d[:4].tolist(),'score':float(d[4])} for i,d in enumerate(caches[frame])],
               'seconds':seconds,'time_block':int(seconds//5),'angle_deg':angle,
               'rgb_status':quality[frame]['status'],'methods':methods}
            q['disagreement']=len({methods[m]['top1'] for m in METHODS})>1
            pool.append(q)
    # Uniform sample of source-box/frame queries, independent of method outcomes.
    rng=np.random.default_rng(20260927)
    random_indices=rng.choice(len(pool),min(random_n,len(pool)),replace=False).tolist()
    used=set(random_indices)
    hard_indices=[i for i,q in enumerate(pool) if i not in used and q['disagreement']]
    hard_indices=rng.choice(hard_indices,min(hard_n,len(hard_indices)),replace=False).tolist()
    selected=[]
    for cohort,indices in [('random',random_indices),('disagreement',hard_indices)]:
        for i in indices:
            q=pool[i];q['cohort']=cohort
            primary=sorted({q['methods'][m]['top1'] for m in METHODS})
            q['review_order']=rng.permutation(primary).tolist()
            selected.append(q)
    output.mkdir(parents=True);(output/'frames').mkdir()
    wanted={f for q in selected for f in (q['frame'],q['previous_frame'])}
    cap=cv2.VideoCapture(str(BASE/'input/aria_rgb_rectified_first60s.mp4'))
    count=0
    for frame in range(600):
        ok,image=cap.read()
        if not ok or image.shape[:2]!=(1920,2560):raise RuntimeError(f'Bad video frame {frame}')
        if frame in wanted:
            if not cv2.imwrite(str(output/f'frames/{frame:04d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,93]):raise OSError('JPEG write failed')
            count+=1
    cap.release()
    data={'version':1,'queries':selected,'pool_size':len(pool),'excluded':dict(excluded),
          'method_names':{'none':'不补偿','rgb':'原 RGB 补偿','aria':'Aria 旋转补偿'}}
    (output/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    inputs=[BASE/'input/aria_rgb_rectified_first60s.mp4',BASE/'input/frame_timestamps.csv',BASE/'input/manifest.json',
            BASE/'cmc/GMC-Aria.txt',BASE/'cmc/quality.tsv',BASE/'cmc/GMC-Aria.txt.meta.json',ROT/'relative_rotations.npz',ROT/'manifest.json']
    manifest={'version':1,'seed':20260927,'random_queries':len(random_indices),'disagreement_queries':len(hard_indices),
              'pool_size':len(pool),'disagreement_pool_size':sum(q['disagreement'] for q in pool),
              'frames_extracted':count,'dataset_sha256':sha(output/'dataset.json'),'inputs':{str(p):sha(p) for p in inputs},
              'cache_sha256_digest':digest.hexdigest(),'source':str(BASE),'rotation':str(ROT),
              'script_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('core.py')),
              'protocol':['Same cached tracker_input80 detection boxes; no thresholds changed; person only',
                          'Independent adjacent-frame source queries; no recursive tracking, KF, appearance, direction, or training',
                          'All transforms applied to four corners then axis-aligned enclosing box; no clipping',
                          'Distance normalized by fixed source diagonal; all candidates retained; stable detector-index tie break',
                          'Independent uniform source-query audit and separate disagreement diagnostic cohort',
                          'Blind unique top1 questions, up to three extra corrections; no implied negative labels',
                          'RGB baseline is existing VideoStab cache including previous-warp fallback (472/600)',
                          'Aria uses earlier exported VIO with offline <=5ms pose interpolation, not strict causal benchmark',
                          'No human labels supplied at initialization; no accuracy claim before human review',
                          'No RL or automatic model updates; immutable scoring protocol and audit events']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['random_queries','disagreement_queries','pool_size','disagreement_pool_size','frames_extracted']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--random',type=int,default=60);parser.add_argument('--disagreement',type=int,default=30)
    args=parser.parse_args();prepare(args.output,args.random,args.disagreement)
