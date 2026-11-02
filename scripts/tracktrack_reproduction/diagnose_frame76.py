"""Read-only source-cache diagnostic; no changes to official tracker or CVAT."""
import json
from pathlib import Path
import numpy as np
import cv2
import replay_tracktrack_cache as replay
import run_tracktrack_video as runner

BASE = Path('/mnt/e/New_research/outputs/tracktrack_reproduction/aria_164557_first60s_v1')
OUT = BASE / 'diagnostic_frame76'
OUT.mkdir(exist_ok=True)
source, args, caches = replay.read_source(BASE / 'tracking')
api = replay.load_tracker_api(Path('/mnt/e/New_research/tools/TrackTrack'))
import trackers.tracker as core
import trackers.utils as utils

def state(t):
    return dict(id=int(t.track_id), state=int(t.state), last=int(t.end_frame_id),
                observations=len(t.history), score=float(t.score),
                predicted_box=t.x1y1x2y2.tolist(), observed_box=t.box.tolist())

original_assignment = core.iterative_assignment
events = []

def assignment(tracks, high, low, deleted, threshold, p, q, step, frame, d_t=3):
    dets = high + low + deleted
    iou, geom = utils.iou_distance(tracks, dets)
    appearance = utils.cos_distance(tracks, dets)
    confidence = utils.conf_distance(tracks, dets)
    angle = utils.angle_distance(tracks, dets, frame, d_t)
    cost = .5 * geom + .5 * appearance + .1 * confidence + .05 * angle
    cost[:, len(high):len(high)+len(low)] += p
    cost[:, len(high)+len(low):] += q
    cost[iou <= .1] = 1.
    cost = np.clip(cost, 0, 1)
    result = original_assignment(tracks, high, low, deleted, threshold, p, q, step, frame, d_t)
    if 55 <= frame <= 101:
        events.append(dict(frame_cvat=frame-1, tracks=[state(t) for t in tracks],
                           detections=[dict(box=d.box.tolist(),score=float(d.score)) for d in dets],
                           high=len(high),low=len(low),deleted=len(deleted),
                           iou=iou.tolist(),geometry_distance=geom.tolist(),
                           appearance_distance=appearance.tolist(),cost=cost.tolist(),
                           matches=np.asarray(result[0]).tolist(), unmatched_tracks=list(result[1]),
                           unmatched_detections=list(result[2])))
    return result

core.iterative_assignment = assignment
for mode, cmc in [('original', BASE/'cmc/GMC-Aria.txt'),
                  ('identity', BASE/'replay_identity/identity_cmc.txt')]:
    events.clear()
    frame_records = []
    lines = []
    with runner.official_tracker(api, args, OUT/mode, cmc) as tracker:
        for frame, path in enumerate(caches[:101],1):
            a,b,_,_ = replay.read_cache(path,frame)
            before=[state(t) for t in tracker.tracks]
            ts=runner.step_tracker(tracker,a,b)
            kept,rejected=runner.filtered_tracks(ts,args.min_box_area)
            for tid,x,y,w,h,score in kept:
                lines.append(f'{frame},{tid},{round(x,1)},{round(y,1)},{round(w,1)},{round(h,1)},{round(score,2)},-1,-1,-1\n')
            if frame>=55:
                frame_records.append(dict(frame_cvat=frame-1, before=before,
                    after=[state(t) for t in tracker.tracks], output=kept,
                    rejected=rejected, detections=a[:,:6].tolist()))
    expected=(BASE/('tracking' if mode=='original' else 'replay_identity')/'tracks.txt').read_text().splitlines(True)
    expected=''.join(l for l in expected if int(l.split(',')[0])<=101)
    assert ''.join(lines)==expected, f'{mode} replay mismatch'
    (OUT/f'{mode}.json').write_text(json.dumps(dict(prefix_output_exact=True,frames=frame_records,assignments=events)),encoding='utf-8')

# Diagnostic contact sheet: same source pixels, cached primary boxes, one-based
# import ordinal inferred from the confirmed frame50 first object ID 194.
cap=cv2.VideoCapture(str(BASE/'input/aria_rgb_rectified_first60s.mp4'))
ordinals={}
count=0
for f,path in enumerate(caches,1):
    with np.load(path) as c:
        ordinals[f]=count
        count+=len(c['det80'])
tiles=[]
for f in [60,65,70,75,76,77,80,85,90]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f)
    ok,im=cap.read()
    assert ok
    with np.load(caches[f]) as c:
        boxes=c['det80'].copy()
    for j,d in enumerate(boxes):
        x1,y1,x2,y2=map(lambda v:int(round(v)),d[:4])
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,255),3)
        cv2.putText(im,f'{ordinals[f+1]+j+1} {d[4]:.2f}',(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
    # Crop right-side people for legibility; bounds include all target boxes.
    tile=im[700:1500,1000:2400]
    tile=cv2.resize(tile,(700,400))
    cv2.putText(tile,f'CVAT frame {f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    tiles.append(tile)
cap.release()
sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,9,3)],axis=0)
cv2.imwrite(str(OUT/'contact_sheet.jpg'),sheet)
print(json.dumps({'output':str(OUT),'prefix_output_exact_both_modes':True}))
