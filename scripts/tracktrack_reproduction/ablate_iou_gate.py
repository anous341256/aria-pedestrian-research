"""Single-variable 0.10 -> 0.05 TrackTrack IoU gate replay, with immutable inputs."""
import argparse
import collections
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import cv2
import replay_tracktrack_cache as replay
import run_tracktrack_video as runner


def summarize(rows):
    by_id=collections.Counter(int(r[1]) for r in rows)
    return dict(output_boxes=len(rows), output_ids=len(by_id),
                frames_with_output=len({int(r[0]) for r in rows}),
                median_visible_frames_per_id=float(np.median(list(by_id.values()))),
                ids_at_most_10_frames=sum(n<=10 for n in by_id.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-only',action='store_true',help='Render already completed replays without changing their MOT files')
    opts=p.parse_args()
    if opts.review_only:
        report=json.loads((opts.output/'summary.json').read_text())
        rows={name:np.loadtxt(opts.output/name/'tracks.txt',delimiter=',',ndmin=2)[:,:7].tolist()
              for name in ['gate010','gate005']}
        render_review(opts,rows,report)
        return
    if opts.output.exists() and any(opts.output.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    opts.output.mkdir(parents=True,exist_ok=True)
    source,args,caches=replay.read_source(opts.base/'tracking')
    api=replay.load_tracker_api(opts.repo)
    import trackers.tracker as core
    import trackers.utils as utils
    original=core.iterative_assignment
    original_code=inspect.getsource(original)
    needle='cost[iou_sim <= 0.10] = 1.'
    assert original_code.count(needle)==1
    patched_code=original_code.replace(needle,'cost[iou_sim <= 0.05] = 1.')
    namespace=dict(vars(utils))
    exec(compile(patched_code,'<single-variable-iou-gate-0.05>','exec'),namespace)
    patched=namespace['iterative_assignment']
    (opts.output/'iterative_assignment_gate005.py').write_text(patched_code)
    tests=[]
    # Isolate gate boundaries from feature quality and candidate competition.
    for gate,code in [(0.10,original_code),(0.05,patched_code)]:
        for iou in [0.,.05,.075,.10,.11]:
            g=dict(vars(utils))
            g['iou_distance']=lambda a,b,v=iou:(np.array([[v]]),np.array([[1-v]]))
            for name in ['cos_distance','conf_distance']:
                g[name]=lambda *a:np.zeros((1,1))
            g['angle_distance']=lambda *a:np.zeros((1,1))
            exec(compile(code,'<gate-boundary-test>','exec'),g)
            matched=bool(len(g['iterative_assignment']([None],[None],[],[],.7,.2,.4,.05,1)[0]))
            assert matched == (iou>gate)
            tests.append(dict(gate=gate,iou=iou,matched=matched))
    results={}
    raw_rows={}
    input_digests={}
    for name,function in [('gate010',original),('gate005',patched)]:
        core.iterative_assignment=function
        dest=opts.output/name
        rows=[]
        rendered_lines=[]
        frame_diagnostics=[]
        digest=hashlib.sha256()
        cmc=opts.base/'tracking/tracker_work/trackers/cmc'/f'GMC-{runner.SEQUENCE_NAME}.txt'
        with runner.official_tracker(api,args,dest,cmc) as tracker:
            for f,path in enumerate(caches,1):
                a,b,_,cachehash=replay.read_cache(path,f)
                digest.update(f'{f}:{cachehash}\n'.encode())
                ts=runner.step_tracker(tracker,a,b)
                kept,rejected=runner.filtered_tracks(ts,args.min_box_area)
                for tid,x,y,w,h,score in kept:
                    rendered_lines.append(f'{f},{tid},{round(x,1)},{round(y,1)},{round(w,1)},{round(h,1)},{round(score,2)},-1,-1,-1\n')
                    rows.append([f,tid,x,y,w,h,score])
                if 70<=f-1<=85:
                    frame_diagnostics.append(dict(frame_cvat=f-1,output_ids=[int(t[0]) for t in kept],
                        active=[dict(id=int(t.track_id),state=int(t.state),observations=len(t.history),box=t.box.tolist())
                                for t in tracker.tracks if t.state in (0,1)],rejected=rejected))
        mot=''.join(rendered_lines)
        (dest/'tracks.txt').write_text(mot)
        (dest/'frame70_85_states.json').write_text(json.dumps(frame_diagnostics,indent=2))
        if name=='gate010':
            assert mot.encode()==(opts.base/'tracking/tracks.txt').read_bytes(), 'Baseline replay changed!'
        results[name]=summarize(rows)
        results[name]['mot_sha256']=hashlib.sha256(mot.encode()).hexdigest()
        raw_rows[name]=rows
        input_digests[name]=digest.hexdigest()
        print(json.dumps({'mode':name,**results[name]}),flush=True)
    core.iterative_assignment=original
    assert len(set(input_digests.values()))==1
    counts={mode:collections.Counter(int(r[0])-1 for r in rows) for mode,rows in raw_rows.items()}
    report=dict(status='replay_complete',baseline_exact=True,only_change='IoU hard gate <=0.10 -> <=0.05',
                boundary_tests=tests,counts_not_accuracy=True,results=results,
                input_cache_digest=input_digests,tracker_parameters=vars(args),
                original_assignment_sha256=hashlib.sha256(original_code.encode()).hexdigest(),
                patched_assignment_sha256=hashlib.sha256(patched_code.encode()).hexdigest(),
                frame_counts=[dict(frame_cvat=f,gate010=counts['gate010'][f],gate005=counts['gate005'][f]) for f in range(600)])
    (opts.output/'summary.json').write_text(json.dumps(report,indent=2))
    render_review(opts,raw_rows,report)


def render_review(opts,raw_rows,report):
    # Side-by-side visualization from exact same source frame in each panel.
    frames={name:collections.defaultdict(list) for name in raw_rows}
    for name,rows in raw_rows.items():
        for r in rows:
            frames[name][int(r[0])-1].append(r)
    cap=cv2.VideoCapture(str(opts.base/'input/aria_rgb_rectified_first60s.mp4'))
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        ffmpeg='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
        if not Path(ffmpeg).is_file():
            raise RuntimeError('ffmpeg required for review video')
    preview=opts.output/'gate010_vs_gate005.mp4'
    target=str(preview)
    if ffmpeg.endswith('.exe'):
        target=subprocess.check_output(['wslpath','-w',str(preview)]).decode().strip()
    proc=subprocess.Popen([ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r','10',
                           '-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                           '-movflags','+faststart','-n',target],stdin=subprocess.PIPE)
    for f in range(600):
        ok,im=cap.read()
        assert ok
        panels=[]
        for name in ['gate010','gate005']:
            panel=cv2.resize(im,(960,720))
            for _,tid,x,y,w,h,score in frames[name][f]:
                x1,y1,x2,y2=[int(round(v*.375)) for v in [x,y,x+w,y+h]]
                color=(50+(int(tid)*73)%206,50+(int(tid)*151)%206,50+(int(tid)*199)%206)
                cv2.rectangle(panel,(x1,y1),(x2,y2),color,2)
                cv2.putText(panel,str(int(tid)),(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            cv2.rectangle(panel,(0,0),(960,42),(0,0,0),-1)
            cv2.putText(panel,f'IoU gate {"0.10" if name=="gate010" else "0.05"} | frame {f} | boxes {len(frames[name][f])}',
                        (15,29),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
            panels.append(panel)
        joined=np.hstack(panels)
        proc.stdin.write(joined.tobytes())
        if f in [75,76,77,78,200,350]:
            cv2.imwrite(str(opts.output/f'comparison_frame_{f:03d}.jpg'),joined)
    cap.release()
    proc.stdin.close()
    assert proc.wait()==0
    check=cv2.VideoCapture(str(preview))
    n=0
    while check.read()[0]:
        n+=1
    check.release()
    assert n==600
    report['preview_decoded_frames']=n
    report['status']='complete'
    (opts.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'preview':str(preview),'decoded_frames':n}),flush=True)

if __name__=='__main__':
    main()
