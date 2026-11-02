"""Fixed discovered seeds/order, negative-off V6 diagnostic for historical case 04."""
import run as experiment
from run import *
def main():
    primary=OUT;control=OUT/'controls/no_negative';experiment.OUT=control
    cid='04';(control/cid).mkdir(parents=True,exist_ok=True);inv=read_json(primary/cid/'inventory.json');dets=read_json(primary/cid/'detections.json');archive={};logs={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for b in range(1,inv['batches']+1):
            source=primary/cid/f'batch_{b:02d}'/'input.json';entries=read_json(source)['entries'];path=control/cid/f'batch_{b:02d}'
            if (path/'runtime.json').exists():masks,decisions=loadgz(path/'masks_rle.jsonl.gz'),loadgz(path/'decisions.jsonl.gz')
            else:masks,decisions=run_batch(tracker,cid,b,entries,archive,logs,dets,enabled=False)
            add_results(masks,decisions,archive,logs);print('negative-off control',cid,b,flush=True)
    write_json(control/cid/'inventory.json',inv);dumpgz(control/cid/'masks_rle.jsonl.gz',[r for rows in archive.values() for r in rows.values()]);dumpgz(control/cid/'decisions.jsonl.gz',[r for rows in logs.values() for r in rows.values()])
    write_json(control/'protocol.json',dict(clip=cid,selection='04 was selected for historical merged-person failure before this negative-off run.',frozen='Same ten entries, seed boxes/frames, four batches and V6 defaults as incremental run. Only inter-batch negative prompting is disabled; automatic discovery is not repeated.',limitation='One selected diagnostic, no GT; mask-count and overlap differences are not identity accuracy.'))
if __name__=='__main__':main()
