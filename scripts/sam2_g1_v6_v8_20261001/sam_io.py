"""Run both V1 branches with complete candidate, decision, and memory evidence."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE']='1'
import collections
import copy
import gzip
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/sam2_negative_prompts_20261001'))
from pilot import load_adapter,Context,decode,describe_mask,ADAPTER,W,H
from common import read_json,write_json,digest
from tracker_geometry import (DEFAULTS,Motion,appearance_similarity,evaluate_candidate,cluster_candidates,
                  assign_candidates,memory_decision,clean,project)

OUT=ROOT/'outputs/sam2_motion_tracker_v1_20261001'
OLD=ROOT/'outputs/sam2_five_people_20261001'
VIDEO=ROOT/'outputs/sam2_fullvideos_20260927/201129/rgb.mp4'
POSE=ROOT/'outputs/factor_feasibility_expanded_20260928/201129_reference.npz'
ALIGN=POSE.with_name('201129_alignment.json')
BRANCHES=['motion_no_camera','motion_with_camera']

import os
CLIP=ROOT/'outputs/sam2_g1_v6_v8_20261001'/os.environ['G1_CLIP']
OLD=CLIP/'input'
VIDEO=OLD/'rgb.mp4'
POSE=OLD/'reference.npz'
ALIGN=OLD/'alignment.json'
N=read_json(OLD/'metadata.json')['frames']


class Camera:
    def __init__(self,path):
        data=np.load(path)
        self.R=data['R'];self.K=data['K'];self.ki=np.linalg.inv(self.K)
        self.times=(data['times_ns']-data['times_ns'][0])/1e9
        self.anchor=self.R[0].copy()
        self.current_frame=(-1)

    def set_current_frame(self,frame):
        assert frame==self.current_frame+1
        self.current_frame=frame

    def matrices(self,frame,enabled):
        assert 0<=frame<=self.current_frame,'Future camera access is forbidden'
        if not enabled:return np.eye(3),np.eye(3)
        return self.K@self.R[frame].T@self.anchor@self.ki,self.K@self.anchor.T@self.R[frame]@self.ki


def descriptor(mask,features,hsv):
    small=cv2.resize(mask.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)
    weight=cv2.resize(small.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA).ravel()
    vector=weight@features/max(float(weight.sum()),1e-8)
    vector/=max(float(np.linalg.norm(vector)),1e-8)
    cmask=cv2.resize(mask.astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST)
    histogram=cv2.calcHist([hsv],[0,1,2],cmask,[12,4,4],[0,180,0,256,0,256]).ravel()
    histogram=np.sqrt(histogram/max(float(histogram.sum()),1e-8))
    return dict(feature=vector,color=histogram)


def make_candidate(model,pp,state,index,owner,source,search_box,features,hsv):
    points=None
    if search_box is not None:
        coords=np.asarray(search_box,dtype=np.float32).reshape(2,2)
        coords*=np.array([model.image_size/W,model.image_size/H],np.float32)
        points=dict(point_coords=torch.as_tensor(coords[None],device='cuda'),
                    point_labels=torch.tensor([[2,3]],device='cuda',dtype=torch.int64))
    current,sam_outputs,_,_=model._track_step(
        frame_idx=index,is_init_cond_frame=False,current_vision_feats=pp.vision_feats,
        current_vision_pos_embeds=pp.vision_pos_embeds,feat_sizes=pp.feat_sizes,
        point_inputs=points,mask_inputs=None,output_dict=state['memory'],num_frames=index+1,
        track_in_reverse=False,prev_sam_mask_logits=None)
    _,_,ious,low,high,pointer,presence=sam_outputs
    current.update(pred_masks=low,pred_masks_high_res=high,obj_ptr=pointer,object_score_logits=presence)
    mask=(torch.nn.functional.interpolate(low,size=(H,W),mode='bilinear',align_corners=False,
                                          antialias=True)[0,0]>0).cpu().numpy()
    stats=describe_mask(mask)
    result=dict(candidate_id=owner+'_'+source,owner=owner,source=source,mask=mask,
                small_mask=cv2.resize(mask.astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST).astype(bool),
                pixels=stats['pixels'],box=stats['bbox_xyxy'],iou_score=float(ious.max()),
                presence=float(presence.sigmoid().item()),prompt_box=search_box,
                descriptor=descriptor(mask,features,hsv),model_output=current,points=points)
    return result


def memory_frames(memory):
    return dict(references=sorted(int(k)+0 for k in memory['cond_frame_outputs']),
                recent=sorted(int(k)+0 for k in memory['non_cond_frame_outputs']))


def commit_memory(model,pp,state,candidate,index,promote,config):
    output=candidate['model_output']
    model._encode_memory_in_output(pp.vision_feats,pp.feat_sizes,candidate['points'],True,
        output['pred_masks_high_res'],output['object_score_logits'],output)
    compact=dict(maskmem_features=output['maskmem_features'],maskmem_pos_enc=output['maskmem_pos_enc'][-1:],
                 pred_masks=output['pred_masks'],obj_ptr=output['obj_ptr'])
    memory=state['memory']
    if promote:
        memory['cond_frame_outputs'][index]=compact
        state['last_reference']=index+0
        while len(memory['cond_frame_outputs'])>1+config['max_extra_references']:
            old=min(k for k in memory['cond_frame_outputs'] if k!=0)
            del memory['cond_frame_outputs'][old]
    else:
        recent=memory['non_cond_frame_outputs']
        recent[index]=compact
        while len(recent)>config['max_recent_memories']:recent.popitem(last=False)


def export_candidate(candidate,evaluation):
    keys=['candidate_id','owner','source','pixels','box','iou_score','presence','prompt_box']
    return clean({**{k:candidate[k] for k in keys},**evaluation})

