"""Version existing unchanged networks and training implementation for new subsets."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SRC=ROOT/'scripts/transformer_matched97_20261006'
def main():
    for name in ['network.py','verify.py','evaluate.py']:
        (HERE/name).write_bytes((SRC/name).read_bytes())
    shared=(SRC/'shared.py').read_text('utf8')
    shared=shared.replace("VERSION = ROOT/'outputs/transformer_matched97_20261006'","VERSION = ROOT/'outputs/g1_near_distribution_subset_20261006'")
    shared=shared.replace("ARM=os.environ.get('G1_BATCH_ARM','new')","ARM=os.environ.get('G1_SUBSET_ARM','matched')").replace("assert ARM in ['old','new']","assert ARM in ['matched','random']")
    shared=shared.replace("BASELINE = ROOT/'outputs/transformer_expanded_20261005'","BASELINE = ROOT/'outputs/transformer_matched97_20261006/new'")
    shared=shared.replace("DATA = OUT/'data'","DATA = BASELINE/'data'").replace("CAMERA = OUT/'camera'","CAMERA = BASELINE/'camera'")
    shared=shared.replace("REPORT = ROOT/'reports/transformer_matched97_20261006'","REPORT = ROOT/'reports/g1_near_distribution_subset_20261006'")
    (HERE/'shared.py').write_text(shared,'utf8')
    run=(SRC/'run.py').read_text('utf8')
    run=run.replace('from train import balanced,seed_all,tensor,DEVICE','from train import seed_all,tensor,DEVICE\n\ndef balanced(indices,metadata,n,rng):\n    # Both arms use identical uniform-window sampling to preserve subset shares.\n    return rng.choice(np.asarray(indices),n,replace=True)')
    run=run.replace("        legacy=dict(code_hashes,**{'run.py':sha(Path(__file__).with_name('run_before_replay_batch_fix.py'))})\n        assert stored==code_hashes or stored==legacy","        assert stored==code_hashes")
    # No old checkpoint is reused for either subset arm.
    start=run.index("    if fold.get('reuse_old_checkpoint'):")
    end=run.index('    opt=torch.optim.AdamW',start)
    run=run[:start]+run[end:]
    (HERE/'run.py').write_text(run,'utf8')
    verify=(HERE/'verify.py').read_text('utf8')
    verify=verify.replace("for k in ['xy','boxes','H','R','cam','seconds']:changed[k][:,P:]=rng.normal(size=changed[k][:,P:].shape)","for k in ['xy','boxes','H','R','cam','seconds']:changed[k]=changed[k][:,:P].copy()\n    # Future suffix physically absent during this independent inference check.")
    (HERE/'verify.py').write_text(verify,'utf8')
if __name__=='__main__':main()
