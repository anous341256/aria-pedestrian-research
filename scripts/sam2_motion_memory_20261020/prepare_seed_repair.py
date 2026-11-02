"""Remove one visually confirmed non-person prompt, identically in all arms."""
from selection import ROOT,OUT,read,write,digest
def main():
 if (OUT/'seed_review_revision.json').exists():return
 path=OUT/'selection.json';selected=read(path)
 before=digest(path);write(OUT/'selection_before_seed_review.json',selected)
 c=next(c for c in selected if c['id']=='050')
 assert c['source']=='sam2_g1_remaining_all_20261007' and c['clip']=='053'
 removed=[e for e in c['entries'] if e['segment']=='N05_S01'];assert len(removed)==1
 c['entries']=[e for e in c['entries'] if e['segment']!='N05_S01']
 write(path,selected)
 summary=read(OUT/'selection_summary.json')
 write(OUT/'selection_summary_before_seed_review.json',summary)
 summary['segment_instances']=sum(len(c['entries']) for c in selected)
 summary['confirmed_nonperson_seed_removed']=1
 write(OUT/'selection_summary.json',summary)
 write(OUT/'seed_review_revision.json',dict(
   reason='Raw seed context visually shows a foreground fixture/printed panel, not a live pedestrian.',
   evidence='presentation/inspection/050_N05_seed_context.jpg',
   method='Reviewed all 15 selected seed crops with historical mask-quarantine flags; no arm outcome used to classify this seed.',
   retained_flagged_seeds='Other 14 show people or occluded/partial people and remain for failure analysis.',
   before_sha256=before,after_sha256=digest(path),clip='050',removed=removed,
   action='Re-run all four arms for clip050, including all remaining targets and negative-donor propagation; archive original outputs.',
   intervals_unchanged=True,thresholds_unchanged=True))
 print('Removed confirmed nonperson 050/N05_S01; queued full four-arm clip replay.')
if __name__=='__main__':main()
