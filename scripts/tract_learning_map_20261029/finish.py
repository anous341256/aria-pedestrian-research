"""Archive descriptive results, figures and reproducibility manifests."""
from pathlib import Path
import json,hashlib,html,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];V='tract_learning_map_20261029';OUT=ROOT/'outputs'/V;P=OUT/'presentation';REP=ROOT/'reports'/V;REG=ROOT/'experiment_registry/experiments'/V
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ver=read(OUT/'verification.json');aud=read(OUT/'independent_audit.json');an=read(OUT/'analysis_verification.json');summ=read(OUT/'analysis_summary.json');cur=read(OUT/'learning_curves.json')['curves'];features=read(OUT/'edgc_features.json')
    assert ver['passed'] and aud['passed'] and an['passed'] and an['source_files_unchanged']
    def c(arm,group,metric='residual',weight='window',span='full'):
        return next(x for x in cur if (x['arm'],x['group'],x['metric'],x['weight'],x['span'])==(arm,group,metric,weight,span))
    fits=[x for x in ver['fits'] if x['arm']=='expanded_long'];shares=[x['actual_old_draws']/11520*100 for x in fits]
    normmedian=float(np.median([r['D_boxheight_normalized_rho'] for r in aud['D_scale_sensitivity']]))
    lines=['TrACT-inspired G1 学习过程诊断（2026-10-29）','',
      '主要发现：扩充训练更适应新增窗口，但对共同旧训练窗口的拟合变弱；验证残差改善而原独立测试变差，提示训练混合比例和录制间泛化需要分开检查。尚未证明新标签较差、模型容量不足或训练步数不足。', '',
      '1. 范围与复现',
      '沿用 horizon_15_15_g1_20261029：过去1.5s的16点预测未来1.5s的15点，65旧+142新窗口，14录制，五个旧录制划分测试折，种子17/29/43。',
      '原数据组/扩充组30次行人训练从头完整重放240步；15个相机直接复用原权重。原模型结构、录制均衡采样、loss、AdamW、学习率、验证选模均不改变。',
      '每20步记录一次（含0，共13次）：实际采样数、训练loss、每窗口每未来时刻残差及图像误差，保存390个诊断检查点。',
      '地图仅含该折训练窗口；原数据组对新增窗口的曲线是未训练探针，不参与优化/选模。验证曲线独立呈现。没有生成测试学习曲线，测试仅在原选中权重复现时核验。',
      f'30组原选中权重、验证曲线、抽样哈希和测试预测全部逐值复现。记录过程未改变Torch/NumPy/Python RNG。累计训练+诊断+保存约{ver["training_seconds"]:.1f}秒，非部署速度。','',
      '2. 学习曲线：行人残差ADE（像素，先每次训练内窗口均值，再五折×三个种子的15次训练等权；不是测试ADE）',
      '组别 | 原数据训练step20/step240 | 扩充训练step20/step240']
    for group,label in [('old','共同旧训练窗口'),('new','新增窗口（原组未训练，扩充组训练）'),('validation','验证录制')]:
        a=c('original_long',group);b=c('expanded_long',group)
        lines.append(f'{label} | {a["mean"][1]:.2f}/{a["mean"][-1]:.2f} | {b["mean"][1]:.2f}/{b["mean"][-1]:.2f}')
    lines += ['',f'共同旧训练窗口：15/15配对在step240的残差ADE变差；新增窗口：{summ["new_training_better_than_untrained_fits"]}/15配对优于旧模型未训练探针。',
      '这不是“模型完全学不动”：两组都从约220–236px迅速下降。扩充模型对旧窗口在后段停留在较高水平，新窗口总体继续改善。',
      f'记录等权敏感性：旧训练窗口末端原/扩充 {c("original_long","old",weight="recording")["mean"][-1]:.2f}/{c("expanded_long","old",weight="recording")["mean"][-1]:.2f}px；新增探针/训练 {c("original_long","new",weight="recording")["mean"][-1]:.2f}/{c("expanded_long","new",weight="recording")["mean"][-1]:.2f}px，方向一致。',
      f'原验证选中步的旧训练残差：{c("original_long","old")["selected_mean"]:.2f}->{c("expanded_long","old")["selected_mean"]:.2f}px；验证残差：{c("original_long","validation")["selected_mean"]:.2f}->{c("expanded_long","validation")["selected_mean"]:.2f}px。',
      f'但相同选中步的验证最终图像ADE：{c("original_long","validation","image")["selected_mean"]:.2f}->{c("expanded_long","validation","image")["selected_mean"]:.2f}px，并未跟随残差改善。损失/选模残差和最终重投影指标存在差异。',
      '原独立测试的残差70.74->110.29px，最终图像171.95->209.17px。该测试是原实验既有结果，未用来设计地图阈值或修改训练。',
      '训练240末端与验证选中的步是不同检查点，不能把表中训练值当成最终模型测试值。', '',
      '3. 采样与归因',
      f'扩充组240步实际抽样中，旧窗口份额{min(shares):.2f}%–{max(shares):.2f}%，旧窗口抽到{min(x["actual_old_draws"] for x in fits)}–{max(x["actual_old_draws"] for x in fits)}次；原数据组为11520次。',
      '数据组成、旧数据暴露量和优化路径同时改变。观察支持“学习分配变化/泛化差异”作为优先假设，但不能单独判定欠训练、梯度冲突、灾难性遗忘或标签质量问题。本轮从头联合训练，不是先学旧再学新的顺序遗忘实验。',
      '本轮五折验证窗口依次4/4/45/4/8；其中三折仅4窗，且窗口可重叠。验证录制的改善不保证其它录制改善。', '',
      '4. 相对数据地图（仅借鉴TrACT，不是完整复现）',
      '每个fold/seed独立：纵轴step240残差ADE，横轴step20..240误差方差。扩充训练池两个维度的中位数为阈值，同阈值用于配对旧组。',
      'Easy=低末端误差/低波动；Trained=低末端误差/高波动；Hard=高末端误差/低波动；Confusing=高末端误差/高波动。均为相对标签。',
      '原论文以训练期间minFDE构图；本版使用现有固定五次生成均值的残差ADE，排除初始化step0的主方差，并额外保留含0与后半段敏感性。这些均是本项目适配，没有使用原论文数值阈值、原型或对比损失。',
      '类别 | 旧窗口占比 | 新窗口占比（每折/种子组内比例平均）']
    for g in ['Easy','Trained','Hard','Confusing']:lines.append(f'{g} | {summ["groups"]["old"]["mean_fit_proportions"][g]*100:.1f}% | {summ["groups"]["new"]["mean_fit_proportions"][g]*100:.1f}%')
    lines += ['新旧的相对类别比例接近，不等于质量相同；中位数阈值本身强制约半数高/低，不能从这个比例反推真实难例占比。',
      f'把初始化step0纳入方差后，旧/新实例的分类改变{summ["groups"]["old"]["sensitivity_with0_disagreement"]*100:.1f}%/{summ["groups"]["new"]["sensitivity_with0_disagreement"]*100:.1f}%；仅用step120..240则改变{summ["groups"]["old"]["sensitivity_late_disagreement"]*100:.1f}%/{summ["groups"]["new"]["sensitivity_late_disagreement"]*100:.1f}%。分类对测量阶段敏感，不宜直接删Hard/Confusing。',
      '方差混合了学习趋势与震荡，Trained也不保证单调改善；需结合example_curves和完整逐窗曲线。', '',
      '5. EDGC对应（只使用16帧历史；不沿用以前6帧阈值）',
      'E=log框高的OLS增长率/s；D=历史框高中位数/1920；G=旋转补偿后纵向OLS速度/1920/s，向下为正；C=RGB光心历史平移路径长/历史时长，m/s。',
      '与扩充模型step240训练残差ADE的逐fold/seed Spearman相关中位数：']
    for k in 'EDGC':
        q=summ['correlations']['all'][k];lines.append(f'{k}: {q["median"]:+.3f}，范围[{q["range"][0]:+.3f},{q["range"][1]:+.3f}]，正向{q["positive_fits"]}/{q["n"]}。')
    lines += [f'D与像素误差的正关联在按历史框高归一化后变为中位数{normmedian:+.3f}。这是事后尺度敏感性检查；除以D也会引入数学耦合，不能把符号变化解释成独立的距离因果效应。尤其不能说“近人更差”。',
      'E/C在该池和当前模型下没有稳定单调关系，不等于这两个因素无效；G是有符号速度，不是运动幅度。EDGC不是四个相互独立的因果因素。',
      '连续特征的旧/新窗口等权中位数：']
    distribution={}
    for k in 'EDGC':
        distribution[k]={s:np.quantile([r['features'][k] for r in features if r['source']==s],[.1,.5,.9]).tolist() for s in ['old','new']}
        lines.append(f'{k}: 旧{distribution[k]["old"][1]:.5f} / 新{distribution[k]["new"][1]:.5f}（p10/p50/p90见feature_distribution.json）')
    write(OUT/'feature_distribution.json',distribution)
    lines += ['', '6. 下一步建议（本轮未执行）',
      '优先拆分“混合比例”和“更新预算”的对照，并通过录制独立验证选择；不要依据本次测试结果删样本。对持续高误差样本做人工中心/边界审计，不能从学习难度直接判错标。暂不引入原型对比损失。', '',
      '7. 验证与限制',
      f'独立重算EDGC207窗口、4974个fit-window记录方差；30个末端检查点重放残差差{aud["residual_metric_max"]:.3g}px；独立SciPy旋转重投影差{aud["scipy_image_metric_max"]:.3g}px；2个整训练池未来截断重放完全一致。',
      f'{ver["protected_files"]}个原始源文件哈希不变。地图/验证共无效投影射线{sum(x["invalid_monitor_rays"]+x["invalid_validation_rays"] for x in ver["fits"])}个。',
      '4974是重复训练/探针记录，不是4974个独立样本；原始仍为207窗且存在重叠。误差相关系数为描述性结果，不报告显著性或独立因果贡献。',
      '相机复用已训练的离线参考系统，训练人物仍用自动标注，非独立密集人工GT。保留所有原权重、标签与已选基线，无CVAT/PPT修改。', '',
      '参考论文：https://arxiv.org/html/2404.12538v2',
      '复现命令（WSL，项目根目录）：',
      '/opt/cvat-sam2/bin/python scripts/tract_learning_map_20261029/run.py',
      '/opt/aria-vrs/bin/python scripts/tract_learning_map_20261029/analyze.py',
      '/opt/cvat-sam2/bin/python scripts/tract_learning_map_20261029/audit.py',
      '/opt/aria-vrs/bin/python scripts/tract_learning_map_20261029/finish.py']
    REP.mkdir(parents=True,exist_ok=True);(REP/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    images=[('learning_curves_zoom.png','先看：训练后段'),('learning_curves.png','完整学习过程'),('data_map.png','相对数据地图'),('example_curves.png','四类实际曲线'),('group_proportions.png','新旧分组比例'),('edgc_map.png','EDGC与学习误差')]
    intro='已精确复现30次训练。扩充组更适应新增窗口，共同旧窗口拟合变弱。当前证据优先支持检查采样配比和跨录制泛化，不能证明新数据质量差。'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 学习过程诊断</title><style>body{font:17px/1.7 system-ui,"Microsoft YaHei";background:#f4f6f9;color:#243247;max-width:1180px;margin:40px auto;padding:20px}section{background:white;padding:24px;margin:25px 0;border-radius:12px}img{width:100%;height:auto}h1{font-size:30px}summary{cursor:pointer}pre{white-space:pre-wrap;font:15px/1.8 inherit}</style>'
    page+=f'<h1>G1 学习过程诊断</h1><p>{intro}</p><p>1.5秒历史 → 1.5秒未来 · 双分支＋平移特征 · 2026-10-29</p>'
    for filename,title in images:page+=f'<section><h2>{title}</h2><a href="{filename}"><img src="{filename}" alt="{title}"></a></section>'
    page+='<section><details><summary>完整报告与方法限制</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details></section></html>'
    (P/'index.html').write_text(page,encoding='utf-8')
    (P/'START_HERE.txt').write_text(intro+'\n\n打开 index.html 查看全部图与报告。\n原始逐窗口记录：../sample_dynamics.json\n每步逐未来时刻误差：../各arm/fold_X/seed_Y/curves.npz\n蓝色为原数据训练，橙色为加入新G1。新增窗口蓝线为未训练探针。\n',encoding='utf-8')
    REG.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','source_hashes.json','verification.json','analysis_verification.json','independent_audit.json','analysis_summary.json','feature_distribution.json','map_thresholds.json','edgc_features.json','unique_window_map.json','example_windows.json','visual_review.json']:
        shutil.copy2(OUT/name,REG/name)
    external=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='delivery_complete.json':external.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(REG/'external_artifacts.json',external)
    delivery=dict(passed=True,version=V,person_fits=30,camera_retrained=0,diagnostic_checkpoints=len(list(OUT.rglob('step_*.pt'))),
        figures=len(images),protected_sources_unchanged=True,artifacts=len(external),report_sha=sha(REP/'RESULTS.txt'))
    assert delivery['diagnostic_checkpoints']==390
    write(OUT/'delivery_complete.json',delivery);write(REG/'delivery_complete.json',delivery)
    print(json.dumps(delivery),flush=True)
if __name__=='__main__':main()
