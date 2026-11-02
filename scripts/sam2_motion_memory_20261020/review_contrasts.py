"""Locate both directions of propagation disagreement for qualitative review."""
import json
import cv2
from selection import OUT, read, write
from render import load

def main():
    categories = {
        'D_pass_A_fail': ('camera_person', 'visual'),
        'A_pass_D_fail': ('visual', 'camera_person'),
        'C_pass_D_fail': ('camera', 'camera_person'),
        'B_pass_D_fail': ('external_box', 'camera_person'),
    }
    found = {k: [] for k in categories}
    clips = read(OUT/'selection.json')
    for c in clips:
        series = {}
        for arm in ['visual', 'external_box', 'camera', 'camera_person']:
            series[arm] = {}
            for row in load(OUT/c['id']/arm/'decisions.jsonl.gz'):
                if 'raw_candidates' not in row:
                    continue
                raw = next(x for x in row['raw_candidates'] if x['source'] == row['raw_source'])
                series[arm][(row['segment'], row['frame'])] = (row, raw)
        for name, (p, q) in categories.items():
            candidates = []
            for key in series[p].keys() & series[q].keys():
                x, a = series[p][key]
                y, b = series[q][key]
                if a['valid'] and not b['valid']:
                    candidates.append(dict(clip=c['id'], recording=c['recording'],
                        segment=key[0], frame=key[1], video_frame=key[1]-c['lo'],
                        passing_arm=p, failing_arm=q,
                        passing_status=x['status'], failing_status=y['status'],
                        proposal_iou_difference=a['guide_iou']-b['guide_iou'],
                        passing_raw=a, failing_raw=b))
            if candidates:
                found[name].append(max(candidates, key=lambda x:x['proposal_iou_difference']))
    dest = OUT/'presentation/inspection'
    output = {}
    for name, candidates in found.items():
        candidates.sort(key=lambda x:x['proposal_iou_difference'], reverse=True)
        chosen=[]; recordings=set()
        for x in candidates:
            if x['recording'] in recordings:
                continue
            recordings.add(x['recording']); chosen.append(x)
            if len(chosen)==3:
                break
        for i, x in enumerate(chosen, 1):
            cap=cv2.VideoCapture(str(OUT/'presentation/raw'/(x['clip']+'.mp4')))
            cap.set(cv2.CAP_PROP_POS_FRAMES, x['video_frame'])
            ok, image=cap.read(); cap.release(); assert ok
            path=dest/f'{name}_{i}_{x["clip"]}.jpg'
            cv2.imwrite(str(path), image)
            x['image']=str(path.relative_to(OUT))
        output[name]=dict(clips_with_this_disagreement=len(candidates), examples=chosen)
    write(dest/'contrast_review.json', dict(
        note='Pass/fail refers only to the common automatic candidate filter, not independently confirmed accuracy. Selected examples are diagnostic, not representative.',
        categories=output))
    print('Balanced-direction contrast review saved.')

if __name__=='__main__':
    main()
