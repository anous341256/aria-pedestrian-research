"""Reviewed first-frame prompts only; fixed before either tracking run."""
import os,sys
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import cv2,numpy as np,torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cvat_sdk.masks import encode_mask
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
sys.path.insert(0,str(ROOT/'scripts/sam2_negative_prompts_20261001'))
from pilot import load_adapter
from common import write_json,read_json,digest
# Boxes in the 1280x960 first-frame preview, scaled by two for model pixels.
BOXES={
 '01':[[595,403,658,555],[661,417,687,497],[687,413,717,495]],
 '02':[[914,480,1026,755],[754,521,819,669],[808,507,876,655],[645,518,685,626]],
 '03':[[866,478,1062,960],[836,513,900,701],[781,532,825,650],[674,532,719,659]],
 '04':[[516,463,536,537],[534,457,555,538],[486,456,523,558]],
 '05':[[691,536,725,625]],
 '06':[[768,565,791,625],[744,563,767,625]],
 '07':[[808,577,1047,960],[622,632,772,960],[565,644,635,822]],
 '08':[[739,110,965,756],[503,153,697,736]],
 '09':[[739,530,908,960],[632,543,672,650],[662,542,689,627]],
 '10':[[635,517,699,675],[460,490,598,830],[737,489,823,673],[699,508,743,615]]}
NAMES={
'01':['white sleeveless, black trousers','striped top, shorts','gray top, shorts'],
'02':['cap, dark striped top','striped top, shorts','gray top, shorts','white top, black trousers'],
'03':['white sleeveless, shoulder bag','cap, dark striped top','striped top, shorts','white top, black trousers'],
'04':['approaching dark top, blue trousers','approaching light top, shorts','white top walking away'],
'05':['light gray top walking away'],
'06':['right approaching light shirt','left approaching light shirt'],
'07':['navy shirt, light shorts','white shirt, dark trousers','approaching person under umbrella'],
'08':['white T, black trousers','white knit top, gray skirt'],
'09':['black top, light trousers, Nike bag','gray top ahead','dark top ahead'],
'10':['black top, pale pink trousers','approaching blue shirt','approaching white T','approaching black top, light trousers']}
COLORS=[(70,240,80),(255,180,50),(240,80,230),(70,210,250)]
def run():
    tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
    predictor=SAM2ImagePredictor(tracker._predictor)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for cid,boxes in BOXES.items():
            folder=OUT/cid/'input';assert not (folder/'initial_masks.json').exists()
            cap=cv2.VideoCapture(str(folder/'rgb.mp4'));ok,bgr=cap.read();assert ok;cap.release()
            predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));initial={};people=[];vis=bgr.copy()
            for j,box in enumerate(boxes):
                ident=chr(65+j);box=np.asarray(box)*2
                masks,scores,_=predictor.predict(box=box,multimask_output=False)
                mask=masks[0].astype(bool);assert mask.sum()>100
                initial[ident]=encode_mask(mask)
                people.append(dict(id=ident,description=NAMES[cid][j],box_prompt=box.tolist(),sam_score=float(scores[0]),pixels=int(mask.sum())))
                color=COLORS[j];vis[mask]=(.65*vis[mask]+.35*np.array(color)).astype(np.uint8)
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis,contours,-1,color,3)
                cv2.putText(vis,ident,(int(box[0]),int(box[1])-12),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,4)
            write_json(folder/'initial_masks.json',initial)
            write_json(folder/'protocol.json',dict(people=people,model='facebook/sam2.1-hiera-large',frame=0,
                selection='First-frame visual prompts only, A is main candidate, nearby people added when distinguishable. No output-based selection.',
                review_status='pending_visual_review',initial_masks_sha256=digest(folder/'initial_masks.json')))
            cv2.imwrite(str(folder/'seed_review.jpg'),cv2.resize(vis,(1280,960)))
            # A fixed crop containing all prompted people makes small distant masks reviewable.
            bb=np.array(boxes);x1=max(0,int(bb[:,0].min())-35)*2;y1=max(0,int(bb[:,1].min())-35)*2
            x2=min(1280,int(bb[:,2].max())+35)*2;y2=min(960,int(bb[:,3].max())+35)*2
            crop=vis[y1:y2,x1:x2];scale=min(2.,1000/max(crop.shape[:2]))
            cv2.imwrite(str(folder/'seed_detail.jpg'),cv2.resize(crop,None,fx=scale,fy=scale))
            print(cid,people,flush=True)
if __name__=='__main__':run()
