"""Delivery index, unchanged-input manifest and persistent experiment record."""
from shared import *
import collections

def main():
 s=read(OUT/'summary.json');cfg=read(OUT/'protocol.json');ver=read(OUT/'verification.json');geo=read(OUT/'geometry_audit.json');cam=read(OUT/'camera_audit.json');media=read(OUT/'presentation/video_checks.json');assert all(d['passed'] for d in [ver,geo,cam,media])
 rows=s['reviewed'];winner=min(VARIANTS,key=lambda v:rows[v]['window_mean']['ADE15']);recwinner=min(VARIANTS,key=lambda v:rows[v]['recording_equal']['ADE15']);hold=rows['hold_last']['window_mean']['ADE15'];p=OUT/'presentation'
 lines=['# 扩充数据后的六组模型：观看入口','',
 '**[先看新增9例六组同步视频](E:/New_research/outputs/transformer_expanded_20261005/presentation/fixed_new9_six_groups.mp4)**（37.8秒，半速）。白圈为 SAM 参考中心，橙圈为预测中心。','',
 '- [旧9例在重新训练后的同步视频](E:/New_research/outputs/transformer_expanded_20261005/presentation/fixed_old9_six_groups.mp4)',
 '- [新增数据的改善／退化案例](E:/New_research/outputs/transformer_expanded_20261005/presentation/selected_new_gain_loss_six_groups.mp4)：事后选择，不能代表平均水平。',
 '- [完整结果与同一旧测试集前后比较](E:/New_research/reports/transformer_expanded_20261005/RESULTS.md)','',
 '主数据：旧117＋新增严格82＝199窗，11个录制留出测试，三个随机种子。新增82中含3个静止对照，已单独报告。','',
 f'六个学习模型中，窗口平均 ADE 最低为 {NAMES[winner]}：{rows[winner]["window_mean"]["ADE15"]:.2f}px；录制等权最低为 {NAMES[recwinner]}：{rows[recwinner]["recording_equal"]["ADE15"]:.2f}px。保持最后位置的窗口 ADE 为 {hold:.2f}px。','',
 '模型结构与上一轮相同。旧预测结果、旧标签和原选定 baseline 均保留。未来图像和 mask 只用于视频展示，不会在预测开始后回输给网络。']
 (p/'START_HERE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 ag=ROOT/'AGENTS.md';txt=ag.read_text(encoding='utf8');marker='## Approved additional G1 near-person annotation'
 values=', '.join(f'{v[0]} {rows[v]["window_mean"]["ADE15"]:.2f}/{rows[v]["window_mean"]["FDE15"]:.2f}' for v in VARIANTS)
 delta=', '.join(f'{v[0]} {s["old_test_before_after"][v]["delta_ADE"]:+.2f}' for v in VARIANTS)
 record=f'''Completed expanded rerun: 198 models, 199 windows, 11 recording folds,
three seeds, 18 exact reused cameras and 15 new held-out cameras. ADE/FDE px:
{values}. Hold-last {hold:.2f}/{rows['hold_last']['window_mean']['FDE15']:.2f}.
Lowest learned window ADE is {winner}; lowest learned recording-equal ADE is
{recwinner}. Same old117 test ADE differences vs earlier training: {delta}.
Original six exclusion pairs/cameras fixed; expanded validation windows also
affect checkpoint choice. Cohort difficulty differs, so do not compare old117
mean directly with overall199 as a training-data gain. New static3 separated.
All198 checkpoints/594 prediction files replayed, future isolation and independent
geometry checks passed. 108 old models replayed in this runtime; old source
files and selected baseline unchanged. Fixed old9/new9 videos and marked new
gain/loss cases saved. See `reports/transformer_expanded_20261005/RESULTS.md`.
No test-driven tuning or baseline replacement was performed.

'''
 assert marker in txt and record not in txt;txt=txt.replace(marker,record+marker,1);ag.write_text(txt,encoding='utf8')
 protected=read(OUT/'source_hashes.json');changed=[k for k,h in protected.items() if sha(ROOT/k)!=h];assert not changed,changed
 files=list((ROOT/'scripts/transformer_expanded_20261005').glob('*.py'))+list(REPORT.glob('*.md'))+[OUT/n for n in ['protocol.json','folds.json','summary.json','verification.json','geometry_audit.json','camera_audit.json','fixed_cases.json']]+list(DATA.glob('*'))+[p/'START_HERE.md']
 write(OUT/'delivery_manifest.json',dict(version='transformer_expanded_20261005',samples=199,models=198,reused_cameras=18,new_cameras=15,protected_files=len(protected),changed_sources=changed,files={f.relative_to(ROOT).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files if f.is_file()},videos=media['checks']))
 write(OUT/'DELIVERED.json',dict(passed=True,report=str((REPORT/'RESULTS.md').relative_to(ROOT)),review_entry=str((p/'START_HERE.md').relative_to(ROOT)),old_assets_preserved=True))
 print('Delivery index, manifest and experiment memory saved.',flush=True)
if __name__=='__main__':main()
