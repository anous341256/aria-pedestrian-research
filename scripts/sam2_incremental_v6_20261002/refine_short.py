"""Review-discovered two-frame foreground person; preserve the automatic first pass."""
import run as experiment
from run import *
def main():
    primary=OUT;dest=OUT/'reviewed';experiment.OUT=dest;cid='10';(dest/cid).mkdir(parents=True,exist_ok=True)
    archive={};logs={};mainm=loadgz(primary/cid/'masks_rle.jsonl.gz');maind=loadgz(primary/cid/'decisions.jsonl.gz');add_results(mainm,maind,archive,logs)
    # Visual inspection of original frames 0..5: the foreground black shirt appears only at 0,1.
    entry=dict(id='P028',frame=0,box=[1740,844,2560,1920],source='visual_review_short_person_full_visible_body_box',visible_frames=[0,1],excluded_proposals=[3,11])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        masks,decisions=run_batch(tracker,cid,11,[entry],archive,logs,read_json(primary/cid/'detections.json')[:2],enabled=True)
    inv=read_json(primary/cid/'inventory.json');inv['entries'].append(entry);inv['batches']=11;inv['review_note']='One clearly visible two-frame foreground person missed by persistence filtering, added in human-reviewed batch 11. Original automatic pass preserved. Propagation is V6, stopped at visually confirmed exit after frame 1.'
    dumpgz(dest/cid/'masks_rle.jsonl.gz',mainm+masks);dumpgz(dest/cid/'decisions.jsonl.gz',maind+decisions);write_json(dest/cid/'inventory.json',inv)
    write_json(dest/'protocol.json',dict(change='Add clip 10 foreground black-shirt pedestrian on frames 0 and 1; all original 134 automatic tracks unchanged.',original_pass=str(primary),manual_input=entry,scope='Offline annotation correction, not an improvement to autonomous V6 identity tracking.'))
if __name__=='__main__':main()
