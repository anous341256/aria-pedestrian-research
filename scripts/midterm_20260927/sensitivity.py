"""Post-hoc robustness audit. Never changes detections, labels, models or splits.

Conditioning and non-overlap checks are diagnostic, not a tuned evaluation set.
"""
from collections import defaultdict
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, CONFIG, read_json, write_json
from forecast import METHODS, N, NI, predict_past_only
from geometry import background_mask, correspondences, fit_similarity, fit_homography


def statistics(rows):
    return {'windows': len(rows), 'tracks': len({(r['clip'], r['track']) for r in rows}),
            'recordings': sorted({r['recording'] for r in rows}),
            'metrics': {m: {f'{metric}{h}': float(np.mean([r[f'{m}_{metric}{h}'] for r in rows])) if rows else None
                            for metric in ['ADE', 'FDE'] for h in [5, 10, 20]} for m in METHODS},
            'freeze_vs_stationary_win_fraction_2s': float(np.mean([r['rgb_factor_freeze_ADE20'] < r['stationary_ADE20'] for r in rows])) if rows else None,
            'freeze_vs_raw_win_fraction_2s': float(np.mean([r['rgb_factor_freeze_ADE20'] < r['raw_cv_ADE20'] for r in rows])) if rows else None}


def rgb_prefix(clip, detections, cutoff):
    """Recompute a video prefix without loading calibration, MPS or future frames."""
    cap = cv2.VideoCapture(str(clip / 'rgb.mp4'))
    ok, im = cap.read()
    assert ok
    previous = cv2.cvtColor(cv2.resize(im, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
    mask0 = background_mask(previous, np.array(detections[0]).reshape(-1, 5), .5)
    hs = [np.eye(3)]
    for f in range(1, cutoff + 1):
        ok, im = cap.read()
        assert ok
        current = cv2.cvtColor(cv2.resize(im, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
        mask1 = background_mask(current, np.array(detections[f]).reshape(-1, 5), .5)
        a, b, _ = correspondences(previous, current, mask0, mask1, .5)
        H = np.eye(3)
        if len(a) >= 35:
            fit = np.random.default_rng(20260927 + f).permutation(len(a))[:int(len(a) * .7)]
            cv2.setRNGSeed(20260927 + f)
            # Match baseline's RNG sequence exactly; this fit does not use any Aria input.
            fit_similarity(a[fit], b[fit])
            H, _ = fit_homography(a[fit], b[fit])
        hs.append(H)
        previous, mask0 = current, mask1
    cap.release()
    return np.array(hs)


def main():
    out = OUT / 'results/sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_json(OUT / 'results/forecast/windows.json') if r['split'] == 'test']
    nonoverlap = []
    by_track = defaultdict(list)
    for r in rows:
        by_track[r['clip'], r['track']].append(r)
    for rs in by_track.values():
        last_end = -1
        for r in sorted(rs, key=lambda x: x['frame']):
            if r['frame'] - 9 > last_end:
                nonoverlap.append(r)
                last_end = r['frame'] + 20
    groups = {
        'all_test': rows,
        'no_image_edge_contact': [r for r in rows if not r['edge_truncated']],
        'historical_rgb_no_identity_fallback': [r for r in rows if r['history_invalid_pairs'] == 0],
        'past_future_rgb_no_identity_fallback': [r for r in rows if r['history_invalid_pairs'] == 0 and r['future_invalid_pairs'] == 0],
        'no_edge_and_past_future_no_fallback': [r for r in rows if not r['edge_truncated'] and r['history_invalid_pairs'] == 0 and r['future_invalid_pairs'] == 0],
        'nonoverlap_per_track': nonoverlap,
    }
    result = {'protocol': 'POST-HOC diagnostic; original full test scores retained; no parameter or label changes',
              'labels': 'assistant visually prechecked partial labels, not independently user-confirmed gold',
              'edge_rule': 'any of the 30 history+future boxes within 5px of image border',
              'fallback_rule': 'stored RGB homography numerically identity; a genuinely static identity transform is not distinguishable in this old cache',
              'nonoverlap_rule': 'greedy earliest 30-frame windows per track; other tracks may still share time and recording, not independent observations',
              'groups': {}, 'recording_macro': {}}
    for key, rs in groups.items():
        result['groups'][key] = {'all': statistics(rs), 'clips': {s['id']: statistics([r for r in rs if r['clip'] == s['id']])
                                                                  for s in read_json(CONFIG)['clips'] if s['split'] == 'test'}}
    for recording in sorted({r['recording'] for r in rows}):
        rs = [r for r in rows if r['recording'] == recording]
        track_results = [statistics(v) for (c, t), v in by_track.items() if v[0]['recording'] == recording]
        result['recording_macro'][recording] = {'windows_weighted': statistics(rs),
                                               'tracks_equal_weight_ADE20': {m: float(np.mean([v['metrics'][m]['ADE20'] for v in track_results])) for m in METHODS}}
    # Audit available support at each lag; these are different surviving events.
    gaze = []
    for spec in read_json(CONFIG)['clips']:
        clip = OUT / 'clips' / spec['id']
        summary = read_json(clip / 'gaze_reviewed/summary.json')
        gaze.append({'clip': spec['id'], 'counts_at_lags_s': {str(int(k) / 10): v['matched_times'] for k, v in summary['lag_motion_difference'].items()},
                     'spatial_common_support': summary['spatial_common_support']})
    result['reviewed_gaze_support'] = gaze
    write_json(out / 'summary.json', result)
    write_json(out / 'nonoverlap_windows.json', [{'clip': r['clip'], 'track': r['track'], 'cutoff': r['frame']} for r in nonoverlap])
    # Independent RGB-prefix replay at the previously selected supporting example.
    clip = OUT / 'clips/test_150809_080'
    cutoff = 24
    detections = read_json(clip / 'yolo/detections.json')['frames']
    actual = rgb_prefix(clip, detections[:cutoff + 1], cutoff)
    stored = np.load(clip / 'geometry/transforms.npz')['rgb_homography'][:cutoff + 1]
    delta = float(np.max(abs(actual - stored)))
    assert delta < 1e-8, delta
    track = next(t for t in read_json(clip / 'annotation_v1/tracks.json')['tracks'] if t['track_id'] == 2)
    boxes = np.array([r['bbox_xyxy'] for r in track['boxes'] if cutoff - 9 <= r['frame_id'] <= cutoff])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / [2560, 1920]
    hs = N @ actual[-10:] @ NI
    hs[0] = np.eye(3)
    pred = predict_past_only(centers, hs)
    cached = next(r for r in read_json(OUT / 'results/forecast/predictions.json') if r['clip'] == 'test_150809_080' and r['track'] == 2 and r['frame'] == cutoff)
    errors = {m: float(np.max(abs(pred[m] * [2560, 1920] - np.array(cached['predictions'][m])))) for m in pred}
    assert max(errors.values()) < 1e-6, errors
    write_json(out / 'rgb_prefix_replay.json', {'clip': 'test_150809_080', 'cutoff': cutoff, 'frames_read': 25,
                                               'MPS_inputs': [], 'future_images_read': 0,
                                               'maximum_transform_cache_difference': delta,
                                               'maximum_prediction_cache_differences_px': errors,
                                               'limit': 'One representative prefix replay; earlier detector cache and rectification are reused. This is not an exhaustive proof of every module.'})
    shown = ['stationary', 'raw_cv', 'rgb_factor_freeze', 'rgb_factor_cv', 'offline_future_rgb']
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    x = np.arange(len(groups))
    for m in shown:
        ax.plot(x, [result['groups'][k]['all']['metrics'][m]['ADE20'] for k in groups], marker='o', label=m)
    ax.set_xticks(x, [f'{k}\nn={len(groups[k])}' for k in groups], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('2s center ADE (px), pooled selected test windows')
    ax.set_title('POST-HOC sensitivity | same predictions | not independent test samples')
    ax.set_yscale('log'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(out / 'forecast_sensitivity.png', dpi=170)
    plt.close(fig)
    for k, rs in groups.items():
        s = result['groups'][k]['all']
        print(k, len(rs), {m: round(s['metrics'][m]['ADE20'], 2) if rs else None for m in shown}, flush=True)
    print('RGB PREFIX REPLAY', delta, errors, flush=True)


if __name__ == '__main__':
    main()
