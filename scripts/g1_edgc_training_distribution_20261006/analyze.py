"""Describe the existing training source and its sampler, without retraining."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib
import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'g1_edgc_training_distribution_20261006'
OUT = ROOT / 'outputs' / VERSION
REPORT = ROOT / 'reports' / VERSION
BANK = ROOT / 'outputs/transformer_matched97_20261006/new'
INVENTORY = ROOT / 'outputs/g1_factor_inventory_20261006'
ORDER = 'EDGC'
LABELS = {'E': '人物框尺度增长率', 'D': '人物框相对高度', 'G': '旋转补偿后纵向速度', 'C': '相机平移速度'}
SCALE = np.array([1., 100., 100., 1.])
UNITS = ['log(框高) / 秒', '占图像高度 %', '图像高度 % / 秒（向下为正）', '米 / 秒']
OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_quantile(values, weights):
    # Inverse weighted empirical CDF, without interpolating between observations.
    ids = np.flatnonzero(weights > 0)
    ids = ids[np.argsort(values[ids])]
    cdf = np.cumsum(weights[ids]) / weights[ids].sum()
    return values[ids[np.minimum(np.searchsorted(cdf, [.1, .5, .9]), len(ids) - 1)]].tolist()


def nonoverlap(rows, by_person=True):
    keep, last = [], {}
    for row in sorted(rows, key=lambda r: (r['recording'], r['end_s'], r['start_s'], r['key'])):
        key = (row['recording'], row['clip'], row['id']) if by_person else row['recording']
        if row['start_s'] > last.get(key, -1) + .049:
            keep.append(row['key'])
            last[key] = row['end_s']
    return keep


def main():
    protected = [BANK / 'data/samples.json', BANK / 'data/dataset.npz', BANK / 'folds.json',
                 BANK / 'protocol.json', INVENTORY / 'existing_windows.json', INVENTORY / 'thresholds_locked.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    meta = read(BANK / 'data/samples.json')
    folds = read(BANK / 'folds.json')
    inv = {r['key']: r for r in read(INVENTORY / 'existing_windows.json')}
    rows = [inv[m['key']] for m in meta]
    thresholds = read(INVENTORY / 'thresholds_locked.json')['thresholds']
    values = np.array([[r['features'][k] for k in ORDER] for r in rows])
    assert len(meta) == 296 and len(set(m['key'] for m in meta)) == 296 and np.isfinite(values).all()
    for m, r in zip(meta, rows):
        assert all(m[k] == r[k] for k in ['key', 'recording', 'clip', 'id', 'frames'])

    # Verify source identity for every prepared array, not just matching names.
    current = dict(np.load(BANK / 'data/dataset.npz'))
    old = dict(np.load(ROOT / 'outputs/transformer_expanded_v2_20261005/data/dataset.npz'))
    newroot = ROOT / 'outputs/sam2_g1_matched97_20261006/dataset'
    newmeta = read(newroot / 'all_eligible_samples.json')
    newpos = {r['key']: i for i, r in enumerate(newmeta)}
    new = dict(np.load(newroot / 'all_eligible_dataset.npz'))
    for key, arr in current.items():
        assert np.array_equal(arr[:199], old[key][:199], equal_nan=True), key
        positions = [newpos[m['key']] for m in meta[199:]]
        assert np.array_equal(arr[199:], new[key][positions], equal_nan=True), key

    # Direct historical D/E/G replay from current arrays and original timestamps.
    maxdiff = {k: 0. for k in 'DEG'}
    calcache = {}
    for i, m in enumerate(meta):
        cid = m['clip']; n = int(cid)
        if cid not in calcache:
            if n <= 10:
                folder = ROOT / 'outputs/sam2_g1_v6_v8_20261001' / cid
            else:
                version = 'sam2_g1_expansion_20261005' if n <= 21 else 'sam2_g1_expansion_v2_20261005' if n <= 31 else 'sam2_g1_matched97_20261006'
                folder = ROOT / 'outputs' / version / cid
            calcache[cid] = dict(np.load(folder / 'input/reference.npz'))
        ns = calcache[cid]['times_ns'][m['frames'][:6]]
        t = (ns - ns[0]) / 1e9; t -= t.mean()
        boxes = current['boxes'][i, :6].astype(float)
        height = boxes[:, 3] - boxes[:, 1]
        xy = current['xy'][i, :6].astype(float)
        p = np.einsum('nij,nj->ni', current['H'][i, :6].astype(float), np.c_[xy, np.ones(6)])
        p = p[:, :2] / p[:, 2:]
        replay = {'D': np.median(height) / 1920, 'E': np.sum(t * np.log(height)) / np.sum(t*t),
                  'G': np.sum(t * p[:, 1]) / np.sum(t*t) / 1920}
        for k, v in replay.items():
            maxdiff[k] = max(maxdiff[k], abs(v - rows[i]['features'][k]))
    assert max(maxdiff.values()) < 1e-10, maxdiff

    weight = np.zeros(len(meta)); used = set(); foldstats = []
    for f in folds:
        idx = np.array(f['train_indices'])
        assert not set(idx) & (set(f['val_indices']) | set(f['test_indices']))
        assert not {meta[i]['recording'] for i in idx} & {f['test'], f['validation']}
        counts = Counter(meta[i]['recording'] for i in idx)
        w = np.zeros(len(meta))
        for i in idx:
            w[i] = 1 / len(counts) / counts[meta[i]['recording']]
        assert np.isclose(w.sum(), 1)
        weight += w / len(folds); used.update(idx.tolist())
        foldstats.append(dict(fold=f['fold'], windows=len(idx), recordings=len(counts),
                              added=len(f['added_train_indices']), test=f['test'], validation=f['validation'],
                              expected_high_share={k: float(w @ (values[:, j] >= thresholds[k])) for j, k in enumerate(ORDER)}))
    selected = np.array(sorted(used)); mask = weight > 0
    assert set(selected) == set(np.flatnonzero(mask)) and np.isclose(weight.sum(), 1)
    levels = values >= np.array([thresholds[k] for k in ORDER])
    featurestats = {}
    for j, k in enumerate(ORDER):
        z = values[selected, j]
        featurestats[k] = dict(name=LABELS[k], raw_unit=UNITS[j], display_multiplier=float(SCALE[j]),
                              quantiles_p10_p50_p90=np.quantile(z, [.1, .5, .9]).tolist(),
                              expected_sampling_quantiles=weighted_quantile(values[:, j], weight),
                              min=float(z.min()), max=float(z.max()), threshold=thresholds[k],
                              high_count=int(levels[selected, j].sum()),
                              expected_high_share=float(weight @ levels[:, j]),
                              per_fold_expected_high_range=[min(f['expected_high_share'][k] for f in foldstats), max(f['expected_high_share'][k] for f in foldstats)])
    cells = []
    for bits in product([0, 1], repeat=4):
        hit = mask & np.all(levels == bits, axis=1)
        ids = np.flatnonzero(hit)
        cells.append(dict(code=''.join(map(str, bits)), order=ORDER, windows=len(ids),
                          recordings=len({meta[i]['recording'] for i in ids}),
                          expected_sampling_share=float(weight[hit].sum()),
                          keys=[meta[i]['key'] for i in ids]))
    recstats = []
    for rec, n in Counter(meta[i]['recording'] for i in selected).most_common():
        ids = [i for i in selected if meta[i]['recording'] == rec]
        recstats.append(dict(recording=rec, windows=n, expected_sampling_share=float(weight[ids].sum()),
                             median_C=float(np.median(values[ids, 3]))))
    centered = values - np.sum(values * weight[:, None], axis=0)
    cov = centered.T @ (centered * weight[:, None])
    wcorr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    active = [rows[i] for i in selected]
    summary = dict(version=VERSION, bank=str(BANK), feature_order=ORDER, pool_windows=len(meta),
                   training_union=len(selected), unused_keys=[meta[i]['key'] for i in np.flatnonzero(~mask)],
                   clips=len({r['clip'] for r in active}), clip_local_ids=len({(r['clip'], r['id']) for r in active}),
                   recordings=len(recstats), folds=foldstats, feature_statistics=featurestats,
                   sampler='Expected probability: each fold equal; within fold recording uniform, then eligible train window uniform. Not observed finite-batch counts, checkpoint-specific exposure, or the factor-test weights.',
                   feature_scope='Historical six reviewed points only, prior to 40/20/40 augmentation. C reuses audited MPS-derived camera-center speed.',
                   cells=cells, recording_statistics=recstats,
                   correlation_pearson=np.corrcoef(values[selected].T).tolist(),
                   correlation_spearman=spearmanr(values[selected]).statistic.tolist(),
                   correlation_expected_sampling=wcorr.tolist(),
                   same_E_G_level_count=int((levels[selected, 0] == levels[selected, 2]).sum()),
                   C_1_to_1p6_count=int(np.sum((values[selected, 3] >= 1) & (values[selected, 3] <= 1.6))),
                   C_below_0p2_count=int(np.sum(values[selected, 3] < .2)),
                   static_control_count=sum(bool(meta[i]['static_control']) for i in selected),
                   numeric_quality_eligible_count=sum(r['eligible'] for r in active),
                   failed_screen_reasons=dict(Counter(reason for r in active for reason in r['reasons'])),
                   person_nonoverlap_keys=nonoverlap(active), recording_time_nonoverlap_keys=nonoverlap(active, False),
                   audit=dict(all_source_arrays_identical=True, DEG_max_replay_difference=maxdiff,
                              historical_features_only=True, source_hashes=before))
    write(OUT / 'distribution.json', summary)
    write(OUT / 'window_distribution.json', [dict(key=m['key'], recording=m['recording'], source=m['source_version'],
          train_folds=[f['fold'] for f in folds if i in f['train_indices']],
          expected_sampling_probability=float(weight[i]), features=rows[i]['features'],
          levels_EDGC=[int(v) for v in levels[i]], prior_inventory_eligible=rows[i]['eligible']) for i, m in enumerate(meta)])

    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, (k, ax) in enumerate(zip(ORDER, axes.flat)):
        z = values[:, j] * SCALE[j]
        # Full observed range retained: no discarded tails or hidden outliers.
        bins = np.linspace(z[mask].min(), z[mask].max(), 24)
        ax.hist(z[mask], bins=bins, weights=np.full(mask.sum(), 100 / mask.sum()), color='#3B6F9C', alpha=.65, label='训练来源窗口占比')
        ax.hist(z[mask], bins=bins, weights=weight[mask] * 100, histtype='step', linewidth=2, color='#D06A2C', label='按现有抽样规则的期望占比')
        ax.axvline(thresholds[k] * SCALE[j], color='#344054', ls='--', lw=1.4, label='先前冻结的高低阈值')
        ax.set(title=f'{k}  {LABELS[k]}', xlabel=UNITS[j], ylabel='该区间占比（%）')
        ax.grid(axis='y', alpha=.16)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('当前训练数据的 E、D、G、C 分布\n295 个实际参与过训练的窗口；11 折抽样期望等权汇总', fontsize=14)
    fig.savefig(OUT / 'marginals.png', dpi=180)
    fig.savefig(OUT / 'marginals.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    codes = ['00', '01', '10', '11']
    matrices = [np.array([[next(c['windows'] for c in cells if c['code'] == a+b) for b in codes] for a in codes]),
                np.array([[next(c['expected_sampling_share'] * 100 for c in cells if c['code'] == a+b) for b in codes] for a in codes])]
    for panel, (ax, matrix) in enumerate(zip(axes, matrices)):
        im = ax.imshow(matrix, cmap='Blues', vmin=0)
        ax.set_xticks(range(4), ['G低 C低', 'G低 C高', 'G高 C低', 'G高 C高'])
        ax.set_yticks(range(4), ['E低 D低', 'E低 D高', 'E高 D低', 'E高 D高'])
        ax.tick_params(axis='both', length=0)
        ax.set_title('各组合实际窗口数' if panel == 0 else '各组合训练抽样期望占比')
        for r in range(4):
            for c in range(4):
                label = str(int(matrix[r,c])) if panel == 0 else f'{matrix[r,c]:.1f}%'
                ax.text(c,r,label,ha='center',va='center',fontsize=13,color='white' if matrix[r,c] > matrix.max() * .55 else '#152238')
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('16 种组合均有样本，但覆盖明显不均\n“高 / 低”沿用原冻结阈值；不表示快速 / 静止、物理近 / 远', fontsize=13)
    fig.savefig(OUT / 'combinations.png', dpi=180)
    fig.savefig(OUT / 'combinations.pdf')
    plt.close(fig)

    report = ['# 当前训练数据的 EDGC 分布', '',
              '范围：最新 `transformer_matched97_20261006/new` 六组行人预测网络。未训练模型、修改抽样器或重新筛选样本；相机网络自身的全录制训练库不在本次人物窗口统计内。', '',
              '## 样本口径', '',
              f'- 准备池296：基础199 + 新增97。11折训练索引并集为{len(selected)}，33段视频、14次录制、78个片段内身份。`51_N01_055`由于各折新增数量匹配，未进入任何训练折。',
              '- 每折实际166–279窗口、12次录制，排除该折测试及验证录制。跨折并集不是某个模型一次训练所用的样本数。',
              '- 最新补标64窗口尚未加入该训练库；214窗口是固定模型因素评价集，不能充当训练样本统计。',
              f'- 按既定贪心规则去重叠后：人物轨迹内{len(summary["person_nonoverlap_keys"])}窗口、录制时间内{len(summary["recording_time_nonoverlap_keys"])}窗口。两者均不等同于独立样本数。', '',
              '## 四维边际分布', '',
              '下表按实际参与过训练的295个来源窗口等权；区间为第10–90百分位，覆盖中间约80%。特征只取过去6点。', '',
              '| 特征 | 中位数 | 第10–90百分位 | 冻结高低阈值 | 高组窗口 | 训练抽样期望高组占比 |',
              '|---|---:|---:|---:|---:|---:|']
    for j,k in enumerate(ORDER):
        stat = featurestats[k]; q=np.array(stat['quantiles_p10_p50_p90'])*SCALE[j]
        report.append(f'| {k} {LABELS[k]}（{UNITS[j]}） | {q[1]:.3f} | {q[0]:.3f}–{q[2]:.3f} | {thresholds[k]*SCALE[j]:.3f} | {stat["high_count"]}/{len(selected)} | {100*stat["expected_high_share"]:.1f}% |')
    report += ['', '- E是框高对数随时间的斜率；正为变大、负为缩小，不等于物理接近速度，也受姿态及标注变化影响。',
               '- D是历史框高中位数/图像高度，只表示表观尺度，不是米制距离。',
               '- G是旋转补偿后的纵向有符号速度，向下为正，仍含平移、视差和标注变化，不能称为纯行人运动。',
               '- C是Aria离线MPS与相机外参导出的相机光心平移路程/时间，不等于行人速度或佩戴者身体速度。', '',
               '## 联合分布与录制集中度', '',
               f'- E与G的Pearson相关为{summary["correlation_pearson"][0][2]:.3f}，Spearman为{summary["correlation_spearman"][0][2]:.3f}，抽样期望加权Pearson为{wcorr[0,2]:.3f}。按高低阈值同组的窗口有{summary["same_E_G_level_count"]}/295。这是共现关系，不是因果或同一运动机制的证明。',
               '- 16种EDGC高低组合均有样本；每格2–53窗口、2–8次录制。E高/D高/G低/C低仅2窗口，期望抽样占比0.18%；E低/D高/G低/C低有53窗口，占17.12%。',
               '- E高/D低/G高/C高为25窗口，期望抽样占比19.25%，是抽样意义下最多的一格。录制均衡并不自动产生特征组合均衡。',
               f'- C在1.0–1.6米/秒的窗口有{summary["C_1_to_1p6_count"]}/295（{summary["C_1_to_1p6_count"]/295:.1%}），低于0.2米/秒只有{summary["C_below_0p2_count"]}窗口。“C低”阈值1.317米/秒仍主要是步行，不等于静止。',
               '- 原始窗口最多的三次录制共188/295（63.7%），其中192702占101/295（34.2%）。现有按录制均衡采样后，各录制期望份额6.1%–8.3%；拥有2个窗口的170022也占8.3%，意味着小录制被重复利用，不能视为数据多样性增加。', '',
               '## 抽样与适用范围', '',
               '训练抽样期望按原算法计算：每折先等概率选择录制，再等概率选择该录制内可训练窗口，最后11折等权汇总。不是已记录的有限批次抽样次数，也不是某个最优检查点实际见过的累计样本；六架构、三个种子不当作新增独立数据。因素对照实验所用的近正交统计权重没有用于这里。', '',
               '分布描述采用修订后历史标注，训练仍有40%修订输入、20%修订前输入、40%噪声增强的混合；没有把图中的干净特征说成所有增强后网络token的分布。EDGC是分析特征，不表示现有网络已经显式输入了这四个量。', '',
               f'按后来固定的更严格G1及标注质量筛选，实际训练并集只有{summary["numeric_quality_eligible_count"]}/295符合；旧基础199中87个符合，已使用的新增96个全部符合。其余未达后续筛选条件，不能直接归为G2/G3或坏标签。本次如实统计旧模型实际用过的数据，没有悄悄删除。', '',
               '## 每折训练量', '', '| 折 | 窗口 | 新增窗口 | 录制数 |', '|---|---:|---:|---:|']
    report += [f'| {f["fold"]} | {f["windows"]} | {f["added"]} | {f["recordings"]} |' for f in foldstats]
    report += ['', '## 16种组合', '', '| EDGC（0低/1高） | 窗口 | 录制 | 期望抽样占比 |', '|---|---:|---:|---:|']
    report += [f'| {c["code"]} | {c["windows"]} | {c["recordings"]} | {100*c["expected_sampling_share"]:.2f}% |' for c in cells]
    report += ['', '## 核验', '', '所有296窗口的实际训练数组与原特征清单来源逐项一致；D/E/G用当前数组和原历史时间戳重算一致，C沿用已审计的相机光心速度。训练/验证/测试录制边界逐折通过；6个来源文件读前读后SHA256一致。完整统计、逐窗口训练折与抽样概率、两张图及PDF位于同版本outputs目录。', '']
    (REPORT / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    assert all(sha(ROOT / rel) == digest for rel, digest in before.items())
    print(json.dumps(dict(pool=296, trained=len(selected), feature_statistics=featurestats,
                         Pearson_EG=summary['correlation_pearson'][0][2], Spearman_EG=summary['correlation_spearman'][0][2],
                         weighted_EG=float(wcorr[0,2]), same_EG=summary['same_E_G_level_count'],
                         nonoverlap=[len(summary['person_nonoverlap_keys']),len(summary['recording_time_nonoverlap_keys'])],
                         audit=summary['audit']['DEG_max_replay_difference'], report=str(REPORT / 'RESULTS.md')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
