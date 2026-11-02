"""Same-mask forward SAM3 tracking with Roboflow-distributed HF weights."""
import argparse
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import time
import traceback

import torch
import transformers
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from PIL import Image

from shared import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=STOP-START)
    parser.add_argument('--name',default='sam3_seeded_hf')
    args=parser.parse_args()
    assert 1 <= args.frames <= STOP-START
    target=OUT/args.name
    target.mkdir(exist_ok=False)
    checkpoint=ROOT/'models/sam3_roboflow_hf'
    seeds=read_json(OUT/'initial_masks.json')
    frame_manifest=read_json(OUT/'frame_manifest.json')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    started=time.perf_counter()
    try:
        model,loading_info=Sam3TrackerVideoModel.from_pretrained(
            str(checkpoint),local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
        loading_info=json.loads(json.dumps(loading_info,default=lambda x:sorted(x) if isinstance(x,set) else str(x)))
        write_json(target/'weight_loading.json',loading_info)
        assert not loading_info.get('missing_keys'), 'Tracker has uninitialised parameters'
        assert not loading_info.get('mismatched_keys'), 'Checkpoint parameter shape mismatch'
        assert not loading_info.get('error_msgs'), 'Checkpoint loading error'
        model=model.eval().to('cuda')
        processor=Sam3TrackerVideoProcessor.from_pretrained(str(checkpoint),local_files_only=True)
        sessions={i:processor.init_video_session(inference_device='cuda',inference_state_device='cpu',
                        processing_device='cpu',video_storage_device='cpu',dtype=torch.bfloat16)
                  for i in IDS}
        # Only image features are shared. Object prompts, masks and temporal memories stay separate.
        shared_cache=sessions['A'].cache
        shared_cache.inference_state_device='cuda'
        for i,session in sessions.items():
            session.cache=shared_cache
            session.video_height=H;session.video_width=W
            processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=IDS.index(i)+1,
                input_masks=decode(seeds[i]),original_size=(H,W))
        setup_seconds=time.perf_counter()-started
        protocol=dict(model='SAM3 visual-prompt video tracker',implementation='Transformers Sam3TrackerVideoModel',
            provider='Roboflow package 45c999b9eaec5eeee18616bd494a5bc9, user-downloaded files',
            checkpoint=read_json(checkpoint/'manual_weights_provenance.json'),
            transformers=transformers.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            precision='bfloat16 weights and autocast; TF32 allowed',seed=0,
            start=START,stop_exclusive=START+args.frames,fps=FPS,input_width=W,input_height=H,
            source_sha256=read_json(OUT/'protocol.json')['source_sha256'],seed_sha256=digest(OUT/'initial_masks.json'),
            script_sha256=digest(Path(__file__)),
            prompts='Same initial masks A-E; five independent states sharing only current-frame image features; no later corrections',
            memory='Official tracker selection, CPU state storage; explicitly copy generated mask memories to configured CPU state device',
            initial_display='Frame 0 archive contains the exact provided annotation, matching baseline convention; raw model seed-mask IoU recorded separately',
            preprocessing='Original lossless source frames; downloaded processor configuration, 1008 square, mean/std .5; no pose or future inputs',
            threshold='Mask logits > 0; no custom confidence filtering, no identity postprocessing',
            scope='PVS fixed-identity tracking only; no text-based detector or new-person discovery',
            timing='Setup separate; tracking includes image IO, preprocessing, propagation, CPU transfer, RLE and verification. Not comparable to legacy tracker-only times.')
        write_json(target/'protocol.json',protocol)
        print('MODEL_READY','missing_keys',len(loading_info.get('missing_keys',[])),
              'unused_keys',len(loading_info.get('unexpected_keys',[])),flush=True)
        torch.cuda.reset_peak_memory_stats()
        track_start=time.perf_counter()
        stats=[]
        seed_iou={}
        with gzip.open(target/'masks_rle.jsonl.gz','wt',encoding='utf-8') as handle, torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for local_idx in range(args.frames):
                source_idx=START+local_idx
                bgr=cv2.imread(str(OUT/frame_manifest[local_idx]['file']))
                assert bgr is not None and bgr.shape==(H,W,3)
                assert hashlib.sha256(bgr.tobytes()).hexdigest()==frame_manifest[local_idx]['decoded_bgr_sha256']
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                inputs=processor(images=Image.fromarray(rgb),device='cpu',return_tensors='pt')
                pixel_values=inputs.pixel_values[0].to(dtype=torch.bfloat16)
                for identity,session in sessions.items():
                    outputs=model(inference_session=session,frame_idx=local_idx,frame=pixel_values,reverse=False)
                    assert outputs.frame_idx==local_idx and outputs.object_ids==[IDS.index(identity)+1]
                    masks=processor.post_process_masks([outputs.pred_masks],original_sizes=inputs.original_sizes,binarize=False)[0]
                    mask=(masks[0,0]>0).cpu().numpy()
                    score=float(outputs.object_score_logits.flatten()[0].float().cpu())
                    assert np.isfinite(score)
                    if local_idx==0:
                        seed=decode(seeds[identity])
                        seed_iou[identity]=float((mask&seed).sum()/max(1,(mask|seed).sum()))
                        rle=seeds[identity]
                    else:
                        rle=encode(mask)
                        assert np.array_equal(decode(rle),mask)
                    area=mask_area(rle)
                    item=dict(frame=source_idx,local_frame=local_idx,id=identity,rle=rle,
                              object_score_logit=score,source='provided_initial_mask' if local_idx==0 else 'model_prediction')
                    handle.write(json.dumps(item,separators=(',',':'))+'\n')
                    stats.append(dict(frame=source_idx,id=identity,area=area,bbox=rle[-4:] if area else None,
                                      object_score_logit=score))
                    # The library's batch memory encoder writes these directly to CUDA.
                    # Honor the configured CPU state storage without pruning or changing values.
                    output_dict=session.output_dict_per_obj[0]
                    storage='cond_frame_outputs' if local_idx==0 else 'non_cond_frame_outputs'
                    stored=output_dict[storage][local_idx]
                    for field in ('maskmem_features','maskmem_pos_enc'):
                        if torch.is_tensor(stored.get(field)):
                            stored[field]=stored[field].to('cpu')
                if local_idx%20==0 or local_idx==args.frames-1:
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-track_start
                    print('TRACKING',local_idx+1,'/',args.frames,'seconds',round(elapsed,1),
                          'gpu_peak_mib',round(torch.cuda.max_memory_allocated()/2**20),flush=True)
        torch.cuda.synchronize()
        tracking_seconds=time.perf_counter()-track_start
        result=dict(status='completed',frames=args.frames,object_frames=len(stats),
                    nonempty_frames={i:sum(r['area']>0 for r in stats if r['id']==i) for i in IDS},
                    setup_seconds=setup_seconds,tracking_seconds=tracking_seconds,
                    peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    raw_model_initial_mask_iou=seed_iou,seed_archive_exact=True,
                    frame_pixel_hashes_verified=True,rle_roundtrips_verified=True,
                    mask_archive_sha256=digest(target/'masks_rle.jsonl.gz'))
        write_json(target/'records.json',stats)
        write_json(target/'runtime.json',result)
        print('RUN_COMPLETE',json.dumps(result),flush=True)
    except Exception as error:
        write_json(target/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
