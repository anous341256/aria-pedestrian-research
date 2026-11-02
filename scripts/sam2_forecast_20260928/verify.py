"""Independent metric recomputation, source preservation and complete video decode."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from prepare import ROOT,SRC,OUT,read,write

def main():
    differences=[];counts={};rows=[]
    for spec in read(OUT/'candidates.json'):
        source=SRC/spec['recording']/'sam2_raw_v1/tracks.json'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_annotation_sha256']
        original=next(t for t in read(source)['tracks'] if t['id']==spec['track'])
        boxes={r['frame']:r['bbox_xyxy'] for r in original['frames'] if r.get('bbox_xyxy')}
        rlist=read(OUT/f'{spec["key"]}_predictions.json')
        assert [r['frame'] for r in rlist]==list(range(spec['start']+5,spec['end']-14))
        for r in rlist:
            f=r['frame'];expected=np.array([boxes[i] for i in range(f+1,f+16)],dtype=float)
            expectedxy=np.c_[(expected[:,:2]+expected[:,2:])/2,expected[:,2:]-expected[:,:2]]
            assert np.array_equal(np.array(r['target_xywh']),expectedxy)
            assert np.array_equal(np.array(r['history_xyxy']),[boxes[i] for i in range(f-5,f+1)])
            for m,pp in r['predictions'].items():
                p=np.array(pp);assert np.isfinite(p).all()
                errs=np.sqrt(((p[:,:2]-expectedxy[:,:2])**2).sum(axis=1))
                for h in [5,10,15]:
                    for metric,value in [('ADE',errs[:h].mean()),('FDE',errs[h-1])]:
                        differences.append(abs(value-r['metrics'][m][f'{metric}_{h/10:.1f}s']))
            if r['evaluated']:rows.append(r)
        counts[spec['key']]=len(rlist)
    error=max(differences);assert error<.002,error
    stored=read(OUT/'summary.json')['all'];assert stored['windows']==len(rows)
    for m in rows[0]['metrics']:
        for metric,value in stored['methods'][m].items():
            assert abs(value-np.mean([r['metrics'][m][metric] for r in rows]))<1e-8
    expected={r['file']:r['frames'] for r in read(OUT/'video_manifest.json')}
    expected['00_how_prediction_works.mp4']=100
    video_results=[]
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    for name,n in expected.items():
        p=OUT/name;cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS)
        header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));i=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(800,1280,3)
            if name=='00_how_prediction_works.mp4' and i in [25,75]:cv2.imwrite(str(qa/f'teaching_{i}.jpg'),im)
            i+=1
        cap.release();assert i==n==header,(name,i,n,header);assert abs(fps-10)<1e-6
        video_results.append(dict(file=name,decoded_frames=i,fps=fps,size=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        print('DECODE_OK',name,i,flush=True)
    checkpoint=ROOT/'models/controlled_baselines/b0_zeroego_seed1/best.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==read(OUT/'verification.json')['checkpoint_sha256']
    write(OUT/'package_verification.json',dict(source_annotation_hashes_unchanged=True,
        checkpoint_hash_unchanged=True,history_target_alignment=True,
        independent_float64_metric_max_difference_pixels=error,summary_recomputed=True,
        scored_windows=len(rows),rolling_counts=counts,videos=video_results,
        scope='Checks file integrity, causal interface and math, not annotation accuracy.'))

if __name__=='__main__':main()
