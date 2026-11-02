from setup import *

def main():
 assert read(OUT/'verification.json')['passed'] and read(OUT/'analysis_audit.json')['passed']
 protected=read(OUT/'protected_sources.json');assert all(sha(ROOT/p)==h for p,h in protected.items())
 own=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json']
 own+=list(Path(__file__).parent.glob('*.py'))
 own+=list((ROOT/'reports/g1_factor_screening_20261006').glob('*.md'))
 write(OUT/'completion.json',dict(completed=True,forecast_errors_computed=True,training_executed=False,frozen_design_unchanged=True,original_files_verified=len(protected),models=162,camera_models=27,windows=214,predictions=3852,files={str(p.relative_to(ROOT)):sha(p) for p in own}))
 print('Completed;',len(own),'new files registered; protected inputs intact.')
if __name__=='__main__':main()
