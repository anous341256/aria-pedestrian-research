"""Bounded, versioned SAM 2 negative-point annotation pilot.

No source pixels are altered and no final masks are subtracted. The intervention
is passed to the SAM prompt encoder as label-0 clicks on reviewed donor regions.
"""
import argparse
import collections
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HOME'] = '/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import decode_mask, encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import fill_holes_in_mask_scores

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/midterm_20260927'))
from common import read_json, write_json, digest, encoder, finish_encoder
OUT = ROOT / 'outputs/sam2_negative_prompts_20261001'
ADAPTER = ROOT / 'tools/cvat-models-official/tracker/sam2/func.py'
W, H = 2560, 1920
CASES = [
    dict(name='food_overlap', clip='outputs/sam2_scene_trial_20260927/162538_020_040',
         start=0, stop=140, source_offset=200, target_id=3, donor_id=1,
         prompt_frame=10, box=[1120, 1080, 1580, 1920],
         negative_frames=[103, 105, 108, 111],
         target='B: long hair, vertical-striped loose shirt, cloth bag',
         donor='A: ponytail with pale hair accessory, pale blue short-sleeve shirt',
         historical_issue='B mask includes part of A hair around local frames 103-111'),
    dict(name='night_identity', clip='outputs/sam2_fullvideos_20260927/201129',
         start=100, stop=340, source_offset=0, target_id=1, donor_id=2,
         prompt_frame=100, box=[928, 1020, 1352, 1920],
         negative_frames=[160, 170, 180, 190, 200, 210, 220],
         target='B: gray checkered long-sleeve shirt, black trousers, black bag',
         donor='A: black short-sleeve shirt with gold print on back',
         historical_issue='B switches to A at 19 s; B reappears at 27 and 33 s'),
]


class Context:
    original_shape_type = 'mask'


def load_adapter():
    spec = importlib.util.spec_from_file_location('negative_pilot_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_masks(case, ids=None, frames=None):
    masks = {}
    archive = ROOT / case['clip'] / 'sam2_raw_v1/masks_rle.jsonl.gz'
    with gzip.open(archive, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            f, oid = row['frame'], row['id']
            if not case['start'] <= f < case['stop']:
                continue
            if ids is not None and oid not in ids:
                continue
            if frames is not None and f not in frames:
                continue
            masks[(f, oid)] = row['rle']
    return masks


def decode(rle):
    return decode_mask(rle, image_width=W, image_height=H)


def choose_negative_points(mask, count=4):
    """Deterministic, separated interior points; every selected frame is reviewed."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((21, 21), np.uint8))
    distance = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    points = []
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(distance)
        if maximum < 5:
            break
        x, y = location
        points.append([x, y])
        cv2.circle(distance, (x, y), 100, 0, -1)
    assert all(mask[y, x] for x, y in points)
    return points


def label(image, lines):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 25 * len(lines) + 9), (0, 0, 0), -1)
    for n, line in enumerate(lines):
        cv2.putText(image, line, (9, 24 + 25*n), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def overlay(image, mask, color):
    result = image.copy()
    small = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    result[small] = (.62 * result[small] + .38 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 1)
    return result


def prepare():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('A frozen protocol already exists. Use a new version.')
    OUT.mkdir(parents=True, exist_ok=True)
    cases = copy.deepcopy(CASES)
    for case in cases:
        dest = OUT / case['name']
        dest.mkdir(exist_ok=True)
        refs = selected_masks(case, [case['donor_id']], case['negative_frames'])
        case['negative_prompts'] = {}
        case['skipped_candidate_frames'] = []
        tiles = []
        cap = cv2.VideoCapture(str(ROOT / case['clip'] / 'rgb.mp4'))
        for f in case['negative_frames']:
            mask = decode(refs[(f, case['donor_id'])])
            points = choose_negative_points(mask)
            if not points:
                case['skipped_candidate_frames'].append(dict(frame=f,
                    reason='No sufficiently thick interior in archived donor mask'))
                continue
            case['negative_prompts'][str(f)] = dict(
                points=points, labels=[0]*len(points),
                source='eroded interior of archived donor mask',
                assistant_points_reviewed=False, user_confirmed=False)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            assert ok and bgr.shape[:2] == (H, W)
            left = cv2.resize(bgr, (640, 480))
            right = overlay(left, mask, (255, 200, 0))
            for i, (x, y) in enumerate(points):
                for panel in (left, right):
                    cv2.circle(panel, (round(x/4), round(y/4)), 5, (0, 0, 255), -1)
                    cv2.putText(panel, str(i+1), (round(x/4)+7, round(y/4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            source_f = f + case['source_offset']
            tiles.append(np.hstack([
                label(left, [f'{case["name"]} f={f}, source {source_f/10:.1f}s',
                             'Red points must belong to A, never B']),
                label(right, ['Archived A donor mask + proposed negative points',
                              'Assistant review required; not user gold'])]))
        cap.release()
        case['planned_negative_frames'] = case['negative_frames']
        case['negative_frames'] = [int(f) for f in case['negative_prompts']]
        for i in range(0, len(tiles), 3):
            cv2.imwrite(str(dest / f'prompt_review_{i//3:02d}.jpg'), np.vstack(tiles[i:i+3]),
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
        case['source_hashes'] = {
            'rgb': digest(ROOT / case['clip'] / 'rgb.mp4'),
            'masks': digest(ROOT / case['clip'] / 'sam2_raw_v1/masks_rle.jsonl.gz')}
    write_json(OUT / 'protocol.json', dict(
        status='awaiting_assistant_point_review', model='facebook/sam2.1-hiera-large',
        cases=cases, variants=['baseline', 'negative_points'],
        comparison='Same RGB, model, initial box/mask and bounded streaming adapter. '
                   'Only intervention: reviewed label-0 prompts and their correction memory.',
        memory_policy='Initial object mask plus at most one latest correction conditioning '
                      'frame; up to model.num_maskmem recent automatic memories.',
        hypothesis='Reviewed other-person pixels as negatives reduce identity/region contamination.',
        selection='Two known historical failures chosen before new inference; diagnostic pilot, '
                  'not an unbiased accuracy benchmark.',
        evaluation='Assistant visual review plus donor-overlap diagnostics. No pixel gold '
                   'or IDF1/accuracy claim; donor overlap is not target correctness.',
        constraints='Original RGB untouched; no post-hoc mask subtraction; no extra positive '
                    'correction, detector, MPS, gaze, new training or CVAT mutation.',
        adapter_sha256=digest(ADAPTER)))
    print('Prepared prompt review sheets and frozen test cases.', flush=True)


def compact_output(out):
    return dict(maskmem_features=out['maskmem_features'],
                maskmem_pos_enc=out['maskmem_pos_enc'][-1:],
                pred_masks=out['pred_masks'], obj_ptr=out['obj_ptr'])


def corrected_step(tracker, pp, state, prompt, use_negative=True):
    """Correct the current predicted logits using genuine SAM label-0 clicks."""
    state.frame_idx += 1
    common = dict(frame_idx=state.frame_idx, is_init_cond_frame=False,
                  current_vision_feats=pp.vision_feats,
                  current_vision_pos_embeds=pp.vision_pos_embeds,
                  feat_sizes=pp.feat_sizes, mask_inputs=None,
                  output_dict=state.predictor_outputs, num_frames=state.frame_idx + 1)
    if use_negative:
        prior = tracker._predictor.track_step(point_inputs=None, run_mem_encoder=False, **common)
        coords = np.array(prompt['points'], dtype=np.float32)
        coords *= np.array([tracker._predictor.image_size/W, tracker._predictor.image_size/H])
        point_inputs = dict(point_coords=torch.as_tensor(coords[None], device='cuda'),
                            point_labels=torch.zeros((1, len(coords)), dtype=torch.int64,
                                                     device='cuda'))
        corrected = tracker._predictor.track_step(
            point_inputs=point_inputs, run_mem_encoder=True,
            prev_sam_mask_logits=prior['pred_masks'].clamp(-32, 32), **common)
    else:
        corrected = tracker._predictor.track_step(point_inputs=None, run_mem_encoder=True, **common)
    corrected['pred_masks'] = fill_holes_in_mask_scores(
        corrected['pred_masks'], tracker._predictor.fill_hole_area)
    out = compact_output(corrected)
    conditioning = state.predictor_outputs['cond_frame_outputs']
    for f in list(conditioning):
        if f != 0:
            del conditioning[f]
    conditioning[state.frame_idx] = out
    # Prior automatic masks can encode the wrong person's identity. Match the
    # documented correction behavior by clearing these temporary memories.
    state.predictor_outputs['non_cond_frame_outputs'].clear()
    mask = torch.nn.functional.interpolate(out['pred_masks'], size=(H, W),
        mode='bilinear', align_corners=False, antialias=True)[0, 0] > 0
    return mask.cpu().numpy()


def describe_mask(mask):
    yy, xx = np.where(mask)
    return dict(pixels=int(len(xx)), bbox_xyxy=None if not len(xx) else
                [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)])


def run_case(tracker, case):
    dest = OUT / case['name']
    if (dest / 'summary.json').exists():
        raise FileExistsError(dest / 'summary.json')
    n = case['stop'] - case['start']
    refs = selected_masks(case, [case['donor_id'], case['target_id']])
    cap = cv2.VideoCapture(str(ROOT / case['clip'] / 'rgb.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, case['start'])
    proc, log = encoder(dest / 'comparison.mp4', 1920, 540, 10, 19)
    states = {}
    records, audit, tiles = [], [], []
    start = time.monotonic()
    page = 0
    with gzip.open(dest / 'masks_rle.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for f in range(case['start'], case['stop']):
            ok, bgr = cap.read()
            assert ok and bgr.shape[:2] == (H, W)
            masks = {v: np.zeros((H, W), bool) for v in ['baseline', 'memory_refresh', 'negative_points']}
            if f >= case['prompt_frame']:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pp = tracker.preprocess_image(Context(), Image.fromarray(rgb))
                if f == case['prompt_frame']:
                    predictor = SAM2ImagePredictor(tracker._predictor)
                    predictor.set_image(rgb)
                    predicted, _, _ = predictor.predict(box=np.array(case['box']),
                                                       multimask_output=False)
                    initial = predicted[0].astype(bool)
                    assert initial.any()
                    for variant in masks:
                        masks[variant] = initial.copy()
                        states[variant] = tracker.init_tracking_state(Context(), pp,
                            TrackableShape(type='mask', points=encode_mask(initial)))
                    del predictor
                else:
                    for variant in masks:
                        prompt = case['negative_prompts'].get(str(f))
                        if variant != 'baseline' and prompt:
                            masks[variant] = corrected_step(tracker, pp, states[variant], prompt,
                                                           use_negative=variant == 'negative_points')
                            if variant == 'negative_points':
                                audit.append(dict(frame=f, points=prompt['points'],
                                              labels=prompt['labels'],
                                              passed_to_sam_prompt_encoder=True))
                        else:
                            shape = tracker.track(Context(), pp, states[variant])
                            if shape:
                                masks[variant] = decode(shape.points)
                del pp
            donor_rle = refs.get((f, case['donor_id']))
            donor = decode(donor_rle) if donor_rle is not None else np.zeros((H, W), bool)
            old_rle = refs.get((f, case['target_id']))
            old = decode(old_rle) if old_rle is not None else np.zeros((H, W), bool)
            row = dict(frame=f, source_frame=f + case['source_offset'],
                       intervention=str(f) in case['negative_prompts'],
                       donor_pixels=int(donor.sum()), donor_is_gold=False,
                       assistant_reviewed_donor_points=str(f) in case['negative_prompts'])
            for variant, mask in masks.items():
                stats = describe_mask(mask)
                intersection = int(np.count_nonzero(mask & donor))
                stats.update(donor_overlap_pixels=intersection,
                    donor_overlap_fraction=intersection / max(1, stats['pixels']))
                row[variant] = stats
                handle.write(json.dumps(dict(frame=f, variant=variant,
                    original_object_id=case['target_id'], rle=encode_mask(mask)),
                    separators=(',', ':')) + '\n')
            row['baseline_vs_archived_iou'] = int(np.count_nonzero(masks['baseline'] & old)) / max(
                1, int(np.count_nonzero(masks['baseline'] | old))) if old.any() or masks['baseline'].any() else 1.0
            row['variants_equal_pixels'] = bool(np.array_equal(masks['baseline'], masks['negative_points']))
            row['memory_control_equal_pixels'] = bool(np.array_equal(masks['baseline'], masks['memory_refresh']))
            records.append(row)
            small = cv2.resize(bgr, (640, 480))
            source_s = (f + case['source_offset']) / 10
            panels = [label(small, ['Original RGB', f'{case["name"]} | {source_s:.1f}s | f={f}'])]
            for variant, title in [('baseline', 'B: original propagation'),
                                   ('negative_points', 'B: with reviewed A negative points')]:
                panel = overlay(small, masks[variant], (0, 220, 255))
                if variant == 'negative_points' and str(f) in case['negative_prompts']:
                    for x, y in case['negative_prompts'][str(f)]['points']:
                        cv2.circle(panel, (round(x/4), round(y/4)), 5, (0, 0, 255), -1)
                panel = label(panel, [title, f'{source_s:.1f}s | target mask in yellow'])
                panels.append(panel)
            combined = np.hstack(panels)
            footer = np.zeros((60, 1920, 3), np.uint8)
            cv2.putText(footer, case['target'], (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .58, (0, 220, 255), 1)
            cv2.putText(footer, 'Diagnostic annotation pilot | original RGB preserved | no pixel gold | no mask subtraction',
                        (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
            canvas = np.vstack([combined, footer])
            proc.stdin.write(canvas.tobytes())
            if f % 10 == 0 or str(f) in case['negative_prompts'] or f == case['stop']-1:
                cv2.imwrite(str(dest / f'frame_{f:04d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if f >= case['prompt_frame']:
                tile = cv2.resize(canvas, (960, 270))
                tiles.append(tile)
                if len(tiles) == 8 or f == case['stop']-1:
                    cv2.imwrite(str(dest / f'review_all_{page:03d}.jpg'), np.vstack(tiles),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    tiles = []
                    page += 1
            if (f-case['start']) % 25 == 0:
                print(case['name'], f-case['start'], '/', n,
                      'seconds', round(time.monotonic()-start, 1), flush=True)
    cap.release()
    finish_encoder(proc, log)
    first = min(case['negative_frames'])
    assert all(r['variants_equal_pixels'] for r in records if r['frame'] < first), 'Branches differ before intervention'
    assert all(r['memory_control_equal_pixels'] for r in records if r['frame'] < first)
    assert len(audit) == len(case['negative_frames'])
    write_json(dest / 'records.json', records)
    write_json(dest / 'applied_negative_prompts.json', audit)
    write_json(dest / 'summary.json', dict(status='inference_complete_pending_visual_review',
        frames=n, negative_prompt_frames=len(audit), negative_points=sum(len(x['points']) for x in audit),
        identical_before_first_intervention=True,
        median_baseline_vs_archived_iou=float(np.median([r['baseline_vs_archived_iou'] for r in records])),
        elapsed_s=time.monotonic()-start, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
        review_pages=page, automatic_not_gold=True))
    del states
    torch.cuda.empty_cache()


def run():
    protocol = read_json(OUT / 'protocol.json')
    assert protocol['status'] == 'assistant_points_reviewed'
    assert digest(ADAPTER) == protocol['adapter_sha256']
    for case in protocol['cases']:
        assert all(p['assistant_points_reviewed'] for p in case['negative_prompts'].values())
    module = load_adapter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        tracker = module.create(model_id=protocol['model'], device='cuda')
        for case in protocol['cases']:
            run_case(tracker, case)
    write_json(OUT / 'runtime.json', dict(torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        protocol_sha256=digest(OUT / 'protocol.json')))
    print('Inference complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    globals()[args.action]()
