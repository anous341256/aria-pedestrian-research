"""Integrity checks and independent diagnostics for the negative-point pilot."""
from collections import Counter
import gzip
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from pilot import OUT, ROOT, W, H, decode, selected_masks, overlay, label
from common import read_json, write_json, digest

if '--negative-only' in sys.argv:
    OUT = OUT / 'negative_only_ablation'
protocol = read_json(OUT / 'protocol.json')
checks = []
diagnostics = {}
for case in protocol['cases']:
    dest = OUT / case['name']
    records = read_json(dest / 'records.json')
    summary = read_json(dest / 'summary.json')
    expected = case['stop'] - case['start']
    source_checks = dict(
        rgb_unchanged=digest(ROOT / case['clip'] / 'rgb.mp4') == case['source_hashes']['rgb'],
        masks_unchanged=digest(ROOT / case['clip'] / 'sam2_raw_v1/masks_rle.jsonl.gz') == case['source_hashes']['masks'])
    assert all(source_checks.values())
    capture = cv2.VideoCapture(str(dest / 'comparison.mp4'))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape[:2] == (540, 1920)
        decoded_frames += 1
    capture.release()
    assert decoded_frames == expected
    expected_by_frame = {r['frame']: r for r in records}
    masks = {}
    counts = Counter()
    picked = ([100, 103, 105, 108, 111, 120, 139] if case['name'] == 'food_overlap'
              else [100, 160, 170, 180, 185, 189, 190, 191, 192, 193, 194, 195, 200, 220, 240, 260, 270, 300, 330, 339])
    with gzip.open(dest / 'masks_rle.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            f, variant = record['frame'], record['variant']
            mask = decode(record['rle'])
            assert mask.shape == (H, W)
            assert int(mask.sum()) == expected_by_frame[f][variant]['pixels']
            counts[variant] += 1
            if f in picked:
                masks[(f, variant)] = mask
    assert dict(counts) == {v: expected for v in protocol['variants']}
    references = selected_masks(case, [case['donor_id']], case['negative_frames'])
    for f, prompt in case['negative_prompts'].items():
        mask = decode(references[(int(f), case['donor_id'])])
        assert prompt['assistant_points_reviewed'] and not prompt['user_confirmed']
        assert all(v == 0 for v in prompt['labels'])
        assert all(mask[y, x] for x, y in prompt['points'])
    cap = cv2.VideoCapture(str(ROOT / case['clip'] / 'rgb.mp4'))
    for f in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, bgr = cap.read()
        assert ok
        small = cv2.resize(bgr, (640, 480))
        panels = [label(small, [f'Original | {case["name"]} | source {(f+case["source_offset"])/10:.1f}s'])]
        for variant, title in [('baseline', 'Original propagation'),
                               ('memory_refresh', 'Memory refresh only, NO negative points'),
                               ('negative_points', 'Memory refresh + negative points')]:
            if variant == 'negative_points' and not protocol.get('negative_memory_refresh', True):
                title = 'Negative points ONLY, original memory policy'
            panels.append(label(overlay(small, masks[(f, variant)], (0, 220, 255)),
                                [title, f'B mask pixels: {expected_by_frame[f][variant]["pixels"]}']))
        canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        cv2.imwrite(str(dest / f'four_way_{f:04d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    first = min(case['negative_frames'])
    after = [r for r in records if r['frame'] >= first]
    before = [r for r in records if case['prompt_frame'] <= r['frame'] < first]
    assert all(r['variants_equal_pixels'] and r['memory_control_equal_pixels'] for r in before)
    table = {}
    for variant in protocol['variants']:
        table[variant] = dict(
            nonempty_frames_after_first_prompt=sum(r[variant]['pixels'] > 0 for r in after),
            frames_overlapping_archived_donor_gt_half_of_target=sum(
                r[variant]['donor_overlap_fraction'] > .5 for r in after),
            total_overlap_pixels_with_archived_donor=sum(r[variant]['donor_overlap_pixels'] for r in after),
            denominator_after_first_prompt=len(after))
    diagnostics[case['name']] = dict(
        note='Overlap is a diagnostic against unreviewed archived masks, NOT identity accuracy. '
             'Empty masks lower overlap without proving target recovery. Prompt frames are included.',
        variants=table,
        selected_frames={str(r['frame']): {v:r[v] for v in protocol['variants']} for r in records if r['frame'] in picked},
        archive_reproduction_median_iou=summary['median_baseline_vs_archived_iou'])
    checks.append(dict(case=case['name'], **source_checks, decoded_video_frames=decoded_frames,
                       mask_records=dict(counts), identical_before_intervention=True,
                       all_points_are_label_zero_inside_reviewed_donor=True))
write_json(OUT / 'verification.json', dict(status='passed', cases=checks))
write_json(OUT / 'diagnostics.json', diagnostics)
print(json.dumps({k: {x:y for x,y in v.items() if x != 'selected_frames'} for k,v in diagnostics.items()}, indent=2))
