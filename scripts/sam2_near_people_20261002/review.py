"""Versioned offline review: revoke identity errors, re-anchor, and complete exits.

All coordinates below were chosen from the local original frames by visual review.
No pixel subtraction, RGB erasure, or interpolation of masks is used.
"""
import run as base
from run import *

ORIGINAL_OUT=OUT
REVOKE={
 '04':{'N01':[2]},
 '07':{'N06':[23,24,25]},
 '08':{'N05':[34,38]},
 '09':{'N04':[7]},
 '10':{'N11':[11,24,41],'N10':[42],'N12':list(range(30,41))},
}
# clip, identity, frame, reviewed box, positive points, reason.
PROMPTS=[
 ('04','N01',2,[1050,994,1100,1165],[[1071,1056]],'separate black-shirt person from adjacent white-shirt person'),
 ('07','N04',27,[0,1140,650,1920],[[200,1370]],'last partial visible frame; absent at frame 28'),
 ('07','N06',26,None,[], 'recheck visible person behind pink foreground person'),
 ('07','N06',27,None,[], 'recheck visible person behind pink foreground person'),
 ('07','N06',28,[340,1200,710,1920],[[500,1430]],'extend visible exit'),
 ('07','N06',29,[120,1120,530,1920],[[280,1370]],'extend visible exit'),
 ('07','N06',30,[0,1100,200,1920],[[70,1390]],'last partial visible frame; absent at frame 31'),
 ('08','N05',38,[1720,474,1820,730],[[1769,532]],'separate background white-shirt man from foreground woman'),
 ('09','N04',7,None,[], 'remove detached foreground-person hand using that persons negative points'),
 ('09','N04',17,[2030,1320,2559,1920],[], 'extend visible exit'),
 ('09','N04',18,[2230,1330,2559,1920],[], 'extend visible exit'),
 ('09','N04',19,[2390,1330,2559,1920],[], 'last partial visible frame; absent at frame 20'),
 ('09','N03',27,[2160,1190,2420,1900],[], 'extend visible exit'),
 ('09','N03',28,[2220,1130,2500,1920],[], 'extend visible exit'),
 ('09','N03',29,[2360,1150,2559,1920],[], 'last partial visible frame; absent at frame 30'),
]

def archive_of(rows):
    archive={}
    for r in rows.values():archive.setdefault(r['id'],{})[r['frame']]=r
    return archive

def main():
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for k in range(1,11):
            cid=f'{k:02d}';folder=ORIGINAL_OUT/cid;dest=folder/'final';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():continue
            rows={(r['id'],r['frame']):r for r in inc.loadgz(folder/'masks_rle.jsonl.gz')}
            logs={(d['id'],d['frame']):d for d in inc.loadgz(folder/'decisions.jsonl.gz')}
            census=read_json(folder/'census.json');events=[]
            for identity,fs in REVOKE.get(cid,{}).items():
                for f in fs:
                    previous=rows.pop((identity,f),None);old=logs.get((identity,f),{})
                    logs[identity,f]=dict(old,status='REVIEW_PENDING',memory_written=False,negative_points=[],prior=None,reason='Visual review: identity error or heavy occlusion; old mask and donation revoked',revision=1)
                    events.append(dict(action='revoke',id=identity,frame=f,had_mask=previous is not None,reason=logs[identity,f]['reason']))
            # No retained result may depend on a revoked donor in this final pass.
            revoked={(i,f) for i,fs in REVOKE.get(cid,{}).items() for f in fs}
            dependencies=[dict(id=d['id'],frame=d['frame'],point=n) for d in logs.values() for n in d.get('negative_points',[]) if (n['donor_id'],n['frame']) in revoked]
            assert not dependencies,dependencies
            selected=[p for p in PROMPTS if p[0]==cid]
            frames=[]
            if selected or cid=='10':
                cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'))
                while True:
                    ok,im=cap.read()
                    if not ok:break
                    frames.append(im)
                cap.release()
            if cid=='10':
                # Re-anchor the counter worker on a genuinely visible frame.
                entry=copy.deepcopy(next(e for e in census['segments'] if e['segment']=='N12_S02'))
                entry.update(start=32,end=38,anchor_frame=35)
                entry['rows']=[r for r in entry['rows'] if 32<=r['frame']<=38]
                corrected={32:[1870,1110,1995,1360],33:[1910,1130,2045,1360]}
                for r in entry['rows']:
                    if r['frame'] in corrected:r.update(box=corrected[r['frame']],source='visual_review_box')
                entry['anchor_box']=next(r['box'] for r in entry['rows'] if r['frame']==35)
                base.OUT=ORIGINAL_OUT/'review_runs'
                rr,dd=run_segment(tracker,cid,entry,frames,np.load(SRC/cid/'input/reference.npz'),archive_of(rows))
                base.OUT=ORIGINAL_OUT
                for r in rr:r.update(revision=1);rows[r['id'],r['frame']]=r
                for d in dd:d.update(status='REVIEW_REANCHOR_'+d['status'],revision=1);logs[d['id'],d['frame']]=d
                original=next(e for e in census['segments'] if e['segment']=='N12_S02')
                original.update(anchor_frame=35,anchor_box=entry['anchor_box'],end_reason='occluded_or_uncertain_after_visible_worker_interval')
                for r in original['rows']:
                    if r['frame'] in corrected:r.update(box=corrected[r['frame']],source='visual_review_box')
                events.append(dict(action='reanchor',id='N12',segment='N12_S02',anchor=35,rerun_interval=[32,38],reason='Original seed was foreground womans head, not counter worker'))
            for _,identity,f,box,positive,reason in selected:
                entry=min((e for e in census['segments'] if e['id']==identity),key=lambda e:min(abs(f-e['start']),abs(f-e['end'])) if not e['start']<=f<=e['end'] else -1)
                guide=next((r for r in entry['rows'] if r['frame']==f),None)
                if box is None:box=guide['box']
                if guide is None:
                    guide=dict(frame=f,box=box,confidence=None,source='visual_review_exit_box',proposal_id=None,detection_id=None);entry['rows'].append(guide);entry['rows'].sort(key=lambda r:r['frame']);entry['end']=max(entry['end'],f)
                else:guide.update(box=box,source='visual_review_prompt_box')
                rgb=cv2.cvtColor(frames[f],cv2.COLOR_BGR2RGB)
                archive=archive_of(rows);donors=donors_at(archive,f,identity,box);negative=inc.near_points(donors,box)
                coords=positive+[n['xy'] for n in negative];labels=[1]*len(positive)+[0]*len(negative)
                predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(rgb)
                kwargs=dict(point_coords=np.array(coords,np.float32),point_labels=np.array(labels,np.int32)) if coords else {}
                mm,ss,_=predictor.predict(box=np.array(box),multimask_output=True,**kwargs)
                choices=[]
                for j,m in enumerate(mm):
                    m=m.astype(bool);q=quality(m,box)
                    conflicts=[dict(donor=d['id'],ratio=float(np.count_nonzero(m & d['mask'])/max(1,min(m.sum(),d['mask'].sum())))) for d in donors]
                    max_overlap=max((c['ratio'] for c in conflicts),default=0.)
                    point_ok=all(m[int(y),int(x)] for x,y in positive)
                    accepted=q['valid'] and float(ss[j])>=.45 and max_overlap<.15 and point_ok
                    choices.append(dict(index=j,sam_score=float(ss[j]),quality=q,conflicts=conflicts,point_ok=point_ok,accepted=accepted))
                good=[c for c in choices if c['accepted']]
                chosen=max(good,key=lambda c:c['sam_score']) if good else None
                rows.pop((identity,f),None)
                log=dict(id=identity,frame=f,segment=entry['segment'],status='REVIEW_PROMPT' if chosen else 'REVIEW_PENDING',reason=reason,revision=1,memory_written=False,prior=None,positive_points=positive,negative_points=negative,prompt_box=box,choices=choices)
                if chosen:
                    m=mm[chosen['index']].astype(bool);q=chosen['quality'];r=dict(id=identity,frame=f,segment=entry['segment'],batch=entry['batch'],rle=encode_mask(m),box=q['box'],quality=q,revision=1,donor_eligible=False)
                    rows[identity,f]=r;log.update(quality=q,sam_score=chosen['sam_score'])
                logs[identity,f]=log;events.append(dict(action='prompt',id=identity,frame=f,accepted=bool(chosen),reason=reason,positive_points=positive,negative_points=negative,box=box))
                print(cid,identity,f,log['status'],[(round(c['sam_score'],3),c['accepted']) for c in choices],flush=True)
            verified_exits={'07':{'N03':20,'N04':27,'N06':30,'N07':40},'08':{'N03':0},'09':{'N04':19,'N03':29},'10':{'N02':11,'N03':17,'N04':32,'N06':1}}
            for e in census['segments']:
                if verified_exits.get(cid,{}).get(e['id'])==e['end']:e['end_reason']='visually_checked_exit_before_next_frame'
                if cid=='02' and e['id']=='N03':e['end_reason']='occluded_by_foreground_person; not_confirmed_exit'
            finalrows=sorted(rows.values(),key=lambda r:(r['frame'],r['id']));finallogs=sorted(logs.values(),key=lambda d:(d['frame'],d['id']))
            inc.dumpgz(dest/'masks_rle.jsonl.gz',finalrows);inc.dumpgz(dest/'decisions.jsonl.gz',finallogs);write_json(dest/'census.json',census)
            write_json(dest/'review.json',dict(events=events,revoked_donor_dependencies=dependencies,source_hashes={n:digest(folder/n) for n in ['census.json','masks_rle.jsonl.gz','decisions.jsonl.gz']},code_sha256=digest(Path(__file__))))
            write_json(dest/'complete.json',dict(rows=len(finalrows),people=census['people'],segments=len(census['segments']),events=len(events)))
            del frames;torch.cuda.empty_cache();print(cid,'FINAL',len(finalrows),flush=True)
if __name__=='__main__':main()
