"""Verify the frozen five-person case and export lossless source frames."""
import hashlib
import shutil
import subprocess

from shared import *


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Frozen protocol already exists; inspect it rather than overwrite.')
    seeds = read_json(SEEDS)
    assert list(seeds) == IDS
    expected = {(f, i) for f in range(START, STOP) for i in IDS}
    seed_masks = {i: decode(seeds[i]) for i in IDS}
    for i, mask in seed_masks.items():
        assert np.array_equal(decode(encode(mask)), mask), i
    seed_overlap_pixels = int((np.sum(list(seed_masks.values()), axis=0) > 1).sum())
    baseline_info = []
    for name, folder, field, value, information in BASELINES:
        path = ROOT / 'outputs' / folder / 'masks_rle.jsonl.gz'
        rows = archive(path, field, value)
        assert set(rows) == expected, (name, len(rows))
        for i in IDS:
            assert np.array_equal(decode(rows[START, i]), seed_masks[i]), (name, i)
        counts = {i: sum(mask_area(rows[f,i]) > 0 for f in range(START, STOP)) for i in IDS}
        baseline_info.append(dict(name=name, archive=str(path.relative_to(ROOT)), field=field,
                                  value=value, information=information, sha256=digest(path),
                                  nonempty_frames=counts, total_frames=STOP-START))
    video_hash = digest(VIDEO)
    original = read_json(ROOT / 'outputs/sam2_five_people_20261001/protocol.json')
    assert video_hash == original['source_sha256'], 'Original source changed'
    shutil.copyfile(SEEDS, OUT / 'initial_masks.json')
    frames_dir = OUT / 'frames'
    frames_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == FPS
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    preview, log = encoder(OUT / 'input_preview.mp4', 1280, 960, FPS)
    frame_manifest = []
    try:
        for source_idx in range(START, STOP):
            ok, frame = cap.read()
            assert ok and frame.shape == (H,W,3), source_idx
            dest = frames_dir / f'{source_idx-START:05d}.png'
            assert cv2.imwrite(str(dest), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            loaded = cv2.imread(str(dest))
            assert np.array_equal(frame, loaded), f'Pixel roundtrip failed {source_idx}'
            frame_manifest.append(dict(frame=source_idx, local_frame=source_idx-START,
                                       file=str(dest.relative_to(OUT)), decoded_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
            preview.stdin.write(cv2.resize(frame,(1280,960),interpolation=cv2.INTER_AREA).tobytes())
            if source_idx == START:
                cv2.imwrite(str(OUT / 'initial_five_people.jpg'), panel(frame,seeds,'Same initial masks for SAM2 and SAM3',source_idx,width=1280))
            if (source_idx-START) % 40 == 0:
                print('EXPORTED',source_idx,flush=True)
    finally:
        cap.release()
        finish_encoder(preview, log)
    write_json(OUT / 'frame_manifest.json', frame_manifest)
    head = subprocess.check_output(['git','-C',str(ROOT/'SAM3'),'rev-parse','HEAD'],text=True).strip()
    protocol = dict(status='input_prepared_sam3_weights_gated_no_sam3_results',
        sam3_repository='https://github.com/facebookresearch/sam3',sam3_commit=head,
        checkpoint='facebook/sam3:sam3.pt',checkpoint_access='Official HF download returned GatedRepoError HTTP 403 on 2026-10-01',
        source=str(VIDEO.relative_to(ROOT)),source_sha256=video_hash,
        source_start=START,source_stop_exclusive=STOP,fps=FPS,width=W,height=H,
        seed_sha256=digest(SEEDS),identities=original['people'],baselines=baseline_info,
        primary_method='SAM3 promptable video tracker, same five initial masks, forward propagation only; no later prompts, camera pose, or reverse propagation.',
        preprocessing='Lossless PNGs from OpenCV-decoded original RGB video; model-specific resize and normalization. Preview MP4 is only for viewing.',
        tracking_mode='Five independent object states sharing the current image features, matching the SAM2 archive. Preserve original seeds including their small boundary overlap.',
        seed_overlap_pixels=seed_overlap_pixels,
        predeclared_review_frames=[100,120,140,160,170,190,194,220,240,267,268,270,300,330,339],
        evidence_limits=[
          'Initial masks were assistant-reviewed, not dense pixel ground truth.',
          'Nonempty frame counts measure output coverage, never accuracy or identity correctness.',
          'Negative-prompt baseline receives additional reviewed information.',
          'Camera variants use offline MPS reference rotation and are not RGB-only.',
          'This single difficult case cannot establish an overall SAM3 superiority claim.',
          'Seeded tracking does not evaluate SAM3 text-prompt detection or automatic object discovery.',
          'Inference and GPU memory compatibility remain untested until official checkpoint access is granted.'
        ])
    write_json(protocol_path,protocol)
    print('PREPARED',str(OUT),flush=True)
    print('BASELINE_NONEMPTY', {b['name']:b['nonempty_frames'] for b in baseline_info},flush=True)


if __name__ == '__main__':
    main()
