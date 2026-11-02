"""Apply an explicit scoped human identity review as a new version; no training.

Old v1 files, predictions and CVAT annotations are not overwritten. Box geometry
remains detector-derived; human identity confirmation is a separate property.
"""
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from common import ROOT, OUT, CONFIG, read_json, write_json, digest
from forecast import windows, predict_past_only, safe_project, METHODS
from gaze_analysis import summarize as summarize_gaze
sys.path.insert(0, str(ROOT / 'tracking'))
from auto_annotate_people import write_cvat

EVENT = Path(__file__).parent / 'reviews/user_identity_20260927_cvat2623956.json'


def main():
    event = read_json(EVENT)
    bundle = OUT / 'annotation_v2'
    assert not bundle.exists(), 'Version already exists; inspect it instead of overwriting review history'
    specs = read_json(CONFIG)['clips']
    original = [OUT / 'annotation_v1' / n for n in ['annotations.xml', 'manifest.json']]
    original += [OUT / 'clips' / s['id'] / 'annotation_v1/tracks.json' for s in specs]
    original += [OUT / 'results/forecast' / n for n in ['summary.json', 'models.npz', 'windows.json', 'predictions.json']]
    before = {str(p): digest(p) for p in original}
    bundle.mkdir(parents=True)
    write_json(bundle / 'review_event.json', event)
    clip_counts = []
    confirmed_global = {}
    confirmed_local = {}
    nconfirmed = nall = 0
    for ci, spec in enumerate(specs):
        src = OUT / 'clips' / spec['id'] / 'annotation_v1/tracks.json'
        payload = deepcopy(read_json(src))
        local_tracks = []
        count = 0
        for t in payload['tracks']:
            selected = []
            for r in t['boxes']:
                matches = [q for q in event['ranges'] if q['clip'] == spec['id'] and q['local_track_id'] == t['track_id']
                           and q['local_start'] <= r['frame_id'] <= q['local_end_inclusive']]
                assert len(matches) <= 1
                yes = bool(matches)
                r.update(human_identity_confirmed=yes, human_box_geometry_confirmed=False,
                         identity_review_event=event['event_id'] if yes else None)
                # Legacy broad approval stays false; only the explicitly named property is approved.
                assert r['human_confirmed'] is False
                if yes:
                    q = matches[0]
                    assert q['global_start'] - q['local_start'] == ci * 100
                    assert q['export_track_id'] == ci * 100 + t['track_id']
                    r['review_status'] = 'user_identity_confirmed_box_geometry_unreviewed'
                    selected.append(deepcopy(r))
                    confirmed_global.setdefault(q['export_track_id'], []).append(dict(r, frame_id=ci*100+r['frame_id'], track_id=q['export_track_id']))
                    count += 1
                nall += 1
            if selected:
                copy = deepcopy(t)
                copy['boxes'] = selected
                local_tracks.append(copy)
        payload['label_status'] = 'mixed: explicit per-node user identity confirmation; all box geometry remains automatic'
        payload['identity_review_event'] = event['event_id']
        payload['legacy_human_confirmed_semantics'] = 'Broad/full-annotation approval; deliberately false. Use human_identity_confirmed for identity only.'
        out = bundle / 'clips' / spec['id']
        out.mkdir(parents=True)
        write_json(out / 'tracks.json', payload)
        if local_tracks:
            confirmed_local[spec['id']] = {'spec': spec, 'tracks': local_tracks,
                                          'label_status': 'user-confirmed identity ONLY; automatic box geometry; not full gold'}
            write_json(out / 'confirmed_identity_only.json', confirmed_local[spec['id']])
        clip_counts.append({'clip': spec['id'], 'tracks': len(payload['tracks']), 'boxes': sum(len(t['boxes']) for t in payload['tracks']), 'user_identity_confirmed_boxes': count})
        nconfirmed += count
    assert nall == 721 and nconfirmed == 108
    assert {k:len(v) for k,v in confirmed_global.items()} == {301:54, 302:54}
    write_cvat(bundle / 'identity_confirmed_only.xml', confirmed_global, OUT / 'annotation_v1/rgb_60s.mp4',
               {'width':2560, 'height':1920, 'fps':10, 'processed_frames':600})
    manifest = {'version':'annotation_v2', 'base_version':'annotation_v1', 'event_id':event['event_id'],
                'review_event_sha256':digest(EVENT), 'clips':clip_counts, 'total_boxes':nall,
                'human_identity_confirmed_boxes':nconfirmed, 'human_identity_confirmed_consecutive_links':106,
                'human_box_geometry_confirmed_boxes':0, 'remaining_assistant_prechecked_boxes':nall-nconfirmed,
                'scope':'ONLY global300–353, CVAT ID7/8, existing nodes. Global354 for ID7 remains unconfirmed.',
                'video':'../annotation_v1/rgb_60s.mp4', 'xml':'identity_confirmed_only.xml contains only the 108 approved-identity nodes; review semantics are in JSON sidecars',
                'no_training':True, 'no_cvat_mutation':True, 'old_assets_sha256':before}
    write_json(bundle / 'manifest.json', manifest)
    results = OUT / 'results/user_identity_confirmed_v2'
    results.mkdir()
    models_npz = np.load(OUT / 'results/forecast/models.npz')
    models = {m:{k:models_npz[m+'_'+k] for k in ['mean','std','coef','bias']} for m in ['ridge_entangled','ridge_factorized']}
    old_rows = {(r['clip'], r['track'], r['frame']):r for r in read_json(OUT / 'results/forecast/windows.json')}
    scored = []
    max_cache_error = 0.
    for clip_id, payload in confirmed_local.items():
        for row in windows(OUT / 'clips' / clip_id, payload['tracks'], 5):
            pred = predict_past_only(row['history'], row['Hs'], models)
            pred['offline_future_rgb'] = np.array([safe_project(p,h) for p,h in zip(pred['rgb_factor_freeze'], row['futureH'])])
            entry = {'clip':clip_id, 'track':row['track'], 'frame':row['frame'], 'history_start':row['frame']-9, 'future_end':row['frame']+20}
            assert 0 <= entry['history_start'] and entry['future_end'] <= 53
            for m, pts in pred.items():
                err = np.linalg.norm((pts-row['target'])*[2560,1920], axis=1)
                for h in [5,10,20]:
                    entry[f'{m}_ADE{h}'] = float(err[:h].mean())
                    entry[f'{m}_FDE{h}'] = float(err[h-1])
                for k in [f'{m}_{q}{h}' for q in ['ADE','FDE'] for h in [5,10,20]]:
                    max_cache_error = max(max_cache_error, abs(entry[k]-old_rows[clip_id,row['track'],row['frame']][k]))
            scored.append(entry)
    assert len(scored) == 10 and max_cache_error < 1e-6
    forecast = {'windows':len(scored), 'recordings':1, 'identities':2, 'models_retrained':False,
                'label_status':'all 30-frame window identity nodes user-confirmed; box geometry still automatic',
                'metrics':{m:{f'{q}{h}':float(np.mean([r[f'{m}_{q}{h}'] for r in scored])) for q in ['ADE','FDE'] for h in [5,10,20]} for m in METHODS},
                'maximum_difference_from_frozen_predictions':max_cache_error,
                'limitations':['post-hoc chosen visible identity subset, not a new independent benchmark', 'overlapping windows from one recording', 'future RGB diagnostic is not deployable', 'no human geometry or completeness validation']}
    write_json(results / 'forecast_windows.json', scored)
    write_json(results / 'forecast_summary.json', forecast)
    association = {}
    for folder in ['association', 'association_unique']:
        all_rows = read_json(OUT / 'results' / folder / 'pairs.json')
        subset = [r for r in all_rows if r['clip'] == 'test_150809_080' and r['track'] in [1,2] and 1 <= r['frame'] <= 53]
        # Assert each scored edge joins two explicitly confirmed unchanged boxes.
        mapping = {t['track_id']:{b['frame_id']:b for b in t['boxes']} for t in confirmed_local['test_150809_080']['tracks']}
        for r in subset:
            assert r['previous_box'] == mapping[r['track']][r['frame']-1]['bbox_xyxy']
            assert r['true_box'] == mapping[r['track']][r['frame']]['bbox_xyxy']
        common = [r for r in subset if r['common_rgb_valid']]
        methods = ['none','aria_rotation','rgb_homography','aria_rgb']
        association[folder] = {'available_pairs':len(subset), 'common_rgb_pairs':len(common),
                               'aria_vs_none_all':{m:float(np.mean([r[m+'_top1'] for r in subset])) for m in methods[:2]} if subset else {},
                               'common_support':{m:float(np.mean([r[m+'_top1'] for r in common])) for m in methods} if common else {}}
        write_json(results / (folder+'_pairs.json'), subset)
    write_json(results / 'association_summary.json', {'results':association, 'protocol':'Same frozen candidate rankings, rescored only on explicitly user-confirmed identity edges.',
                                                       'limitations':'Duplicate-detection ambiguity remains in unrestricted version. Strict version excludes it for all methods. Not full MOT accuracy or box-boundary gold.'})
    gaze = summarize_gaze(confirmed_local['test_150809_080']['spec'], True,
                         tracks_file=bundle / 'clips/test_150809_080/confirmed_identity_only.json',
                         output_subdir='gaze_user_identity_v2',
                         identity_source='user-confirmed ID7/8 global300–353 ONLY; box geometry automatic')
    write_json(results / 'gaze_summary.json', gaze)
    after = {str(p):digest(p) for p in original}
    assert before == after, 'v1 data/model/predictions must remain unchanged'
    xml = ET.parse(bundle / 'identity_confirmed_only.xml').getroot()
    boxes = [b for t in xml.findall('track') for b in t.findall('box') if b.attrib['outside'] == '0']
    assert len(boxes) == 108 and all(300 <= int(b.attrib['frame']) <= 353 for b in boxes)
    verification = {'status':'passed', 'original_v1_unchanged':True, 'models_unchanged':True,
                    'confirmed_identity_nodes':108, 'confirmed_consecutive_links':106, 'box_geometry_confirmed':0,
                    'other_nodes_upgraded':0, 'xml_confirmed_nodes':108, 'confirmed_forecast_windows':10,
                    'prediction_cache_difference':max_cache_error, 'review_event_sha256':digest(EVENT)}
    write_json(results / 'verification.json', verification)
    print('REVIEW APPLIED', verification, flush=True)
    print('FORECAST', {m:round(forecast['metrics'][m]['ADE20'],2) for m in METHODS}, flush=True)
    print('ASSOCIATION', association, flush=True)


if __name__ == '__main__':
    main()
