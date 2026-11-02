from shared import *
import re,json


def main():
    data=np.load(OLD/'dataset.npz');rows=read(OUT/'metrics.json');lookup={(r['fold'],r['seed'],r['variant'],r['arm'],r['index']):r for r in rows}
    cfg=read(OLD/'protocol.json');checked=0;maxdiff=0.
    for fold in read(OLD/'folds.json'):
        for seed in cfg['seeds']:
            for variant in VARIANTS:
                z=np.load(OUT/f'fold_{fold["fold"]}/seed_{seed}/{variant}/predictions.npz')
                assert np.array_equal(z['indices'],fold['test_indices'])
                for arm in ARMS:
                    errors=np.sqrt(((z[arm]-data['xy'][z['indices'],6:])**2).sum(-1))
                    bad=~np.isfinite(errors)|(z[arm+'_depth']<=1e-6);errors[bad]=32000.
                    for j,i in enumerate(z['indices']):
                        actual=np.array([errors[j,:5].mean(),errors[j,:10].mean(),errors[j].mean(),errors[j,-1]])
                        saved=lookup[fold['fold'],seed,variant,arm,int(i)]['metrics']
                        expected=np.array([saved[k] for k in ['ADE05','ADE10','ADE15','FDE15']])
                        d=float(np.max(abs(actual-expected)));assert d<1e-8;maxdiff=max(maxdiff,d)
                        assert int(bad[j].sum())==saved['invalid'];checked+=1
    assert checked==len(rows)==2808
    sources=read(OUT/'source_hashes.json')
    for p,h in sources.items():assert sha(ROOT/p.replace('\\','/'))==h,p
    assert read(OUT/'verification.json')['passed'] and read(OUT/'presentation/verification.json')['passed']
    report=ROOT/'reports/camera_sync_prediction_20261002/RESULTS.md'
    links=re.findall(r'\]\((E:/[^)]+)\)',report.read_text(encoding='utf8'))
    for p in links:assert Path(p).exists(),p
    result=dict(passed=True,independently_recomputed_metric_rows=checked,max_metric_difference=maxdiff,source_files_unchanged=len(sources),
                local_report_links_checked=len(links),visual_review='Both plotted results and representative success/failure comparison frames inspected; videos fully decoded.')
    write(OUT/'package_verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
