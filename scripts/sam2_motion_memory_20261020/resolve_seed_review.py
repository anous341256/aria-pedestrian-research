"""Withdraw a single-frame non-person inference after reviewing temporal context."""
from selection import ROOT,OUT,read,write,digest
def main():
 revision=read(OUT/'seed_review_revision.json')
 write(OUT/'initial_seed_review_assessment.json',revision)
 original=OUT/'selection_before_seed_review.json'
 assert digest(original)==revision['before_sha256']
 (OUT/'selection.json').write_bytes(original.read_bytes())
 (OUT/'selection_summary.json').write_bytes((OUT/'selection_summary_before_seed_review.json').read_bytes())
 write(OUT/'seed_review_revision.json',dict(active=False,decision='withdrawn_no_targets_removed',
   reason='Temporal source frames0/2/7/11 suggest a bent/partly occluded person; the single-frame fixture inference was not reliable.',
   evidence='presentation/inspection/050_seed_context_sequence.jpg',
   target='050/N05_S01',status='retain as uncertain initialization for human review',
   final_selection_sha256=digest(OUT/'selection.json'),
   same_as_initial_selection=digest(OUT/'selection.json')==revision['before_sha256'],
   mask_outputs_modified=False))
 print('Restored exact initial selection. No mask outputs changed or removed.')
if __name__=='__main__':main()
