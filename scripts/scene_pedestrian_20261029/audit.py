"""Read-only shared-data inventory. No new labels, fits or selection by outcomes.
Run with stdlib Python; --data-root is the original shared research checkout.
"""
import argparse, bisect, collections, csv, hashlib, json, math, statistics
from pathlib import Path

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    root=a.data_root; a.out.mkdir(parents=True,exist_ok=True)
    protected={}
    def source(p):
        protected[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return read(p)
    notes=source(root/'scripts/sam2_scene_trial_20260927/scene_notes.json')
    notes={n['id']:n for n in notes}
    bulk=source(root/'outputs/g1_bulk_association_20261029/inventory.json')
    cohorts={
        'bulk_quality':source(root/'outputs/g1_bulk_association_20261029/dataset/quality_metadata.json'),
        'short_expanded':source(root/'outputs/g1_expanded_forecast_20261029/expanded_samples.json'),
        'long_expanded':source(root/'outputs/horizon_15_15_g1_20261029/samples.json')}
    recs=[]; times={}; windows=[]
    for vrs in sorted((root/'data/Aria/raw_vrs').glob('*.vrs')):
        rec=vrs.stem; mps=vrs.parent/f'mps_{rec}_vrs'; gaze=mps/'eye_gaze/general_eye_gaze.csv'
        item={'recording':rec,'vrs_bytes':vrs.stat().st_size,'scene_note':notes.get(rec[-6:],{}),
              'general_gaze':gaze.exists(),'personalized_gaze':(mps/'eye_gaze/personalized_eye_gaze.csv').exists(),
              'closed_loop_pose':(mps/'slam/closed_loop_trajectory.csv').exists(),
              'open_loop_pose':(mps/'slam/open_loop_trajectory.csv').exists()}
        if (mps/'eye_gaze/summary.json').exists():item['gaze_summary']=source(mps/'eye_gaze/summary.json')
        if (mps/'slam/summary.json').exists():
            s=source(mps/'slam/summary.json'); item['slam_info']=s.get('SLAM',{})
        if gaze.exists():
            protected[gaze.relative_to(root).as_posix()]=hashlib.sha256(gaze.read_bytes()).hexdigest()
            with gaze.open() as f:g=list(csv.DictReader(f))
            ts=[int(r['tracking_timestamp_us'])*1000 for r in g]
            finite=[math.isfinite(float(r['yaw_rads_cpf'])) and math.isfinite(float(r['pitch_rads_cpf'])) for r in g]
            dt=[(b-a)/1e6 for a,b in zip(ts,ts[1:])]
            item.update(gaze_rows=len(g),finite_angles=sum(finite),gaze_first_ns=ts[0],gaze_last_ns=ts[-1],
                        gaze_span_s=(ts[-1]-ts[0])/1e9,median_interval_ms=statistics.median(dt),
                        gaps_over_300ms=sum(d>300 for d in dt),nonincreasing_times=sum(d<=0 for d in dt))
            times[rec]=ts
        item['bulk_clips']=sum(c['recording']==rec for c in bulk['clips'])
        for name,cohort in cohorts.items():item[name+'_windows']=sum(m['recording']==rec for m in cohort)
        recs.append(item)
    for name,cohort in cohorts.items():
        for m in cohort:
            # Each metadata timestamp array covers the complete window; use only first16 history.
            ts=times.get(m['recording'],[]); stamps=m.get('timestamps_ns')
            if stamps is None:
                windows.append({'cohort':name,'key':m['key'],'recording':m['recording'],'status':'missing_timestamp_array'})
                continue
            lo,hi=map(int,(stamps[0],stamps[15])); l=bisect.bisect_left(ts,lo); r=bisect.bisect_right(ts,hi)
            chosen=ts[l:r]; assert all(lo<=t<=hi for t in chosen)
            windows.append({'cohort':name,'key':m['key'],'recording':m['recording'],
                'history_start_ns':lo,'cutoff_ns':hi,'native_history_gaze_rows':len(chosen),
                'max_gap_ms':max([(b-a)/1e6 for a,b in zip(chosen,chosen[1:])],default=None),'status':'audited'})
    summary={'raw_recordings':len(recs),'general_gaze_recordings':sum(r['general_gaze'] for r in recs),
             'personalized_recordings':sum(r['personalized_gaze'] for r in recs),
             'closed_loop_recordings':sum(r['closed_loop_pose'] for r in recs),
             'gaze_rows':sum(r.get('gaze_rows',0) for r in recs),
             'summed_gaze_span_s':sum(r.get('gaze_span_s',0) for r in recs),
             'bulk_clips':len(bulk['clips']), 'bulk_frame_entries':len(bulk['entries']),
             'cohorts':{n:{'windows':len(c),'recordings':len({m['recording'] for m in c}),
                 'historical_gaze_count_histogram':dict(collections.Counter(w['native_history_gaze_rows'] for w in windows if w['cohort']==n and w['status']=='audited')),
                 'missing_timestamp_arrays':sum(w['cohort']==n and w['status']!='audited' for w in windows)} for n,c in cohorts.items()},
             'meaning':'Durations are sums of gaze first-to-last spans, not verified usable walking time. Scene notes are recording-level prior descriptions, not event labels. Gaze rows and overlapping windows are not independent people/events.'}
    for fn,x in [('inventory.json',{'summary':summary,'recordings':recs,'windows':windows}),('source_hashes.json',protected)]:
        (a.out/fn).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
