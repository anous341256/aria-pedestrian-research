"""Preserve results, index review media and record completed measurements."""
from shared import *
from datetime import datetime

def main():
 s=read(OUT/'summary.json');ver=read(OUT/'verification.json');geo=read(OUT/'geometry_audit.json');cam=read(OUT/'camera_audit.json');media=read(OUT/'presentation/video_checks.json')
 assert all(a['passed'] for a in [ver,geo,cam,media]);assert (REPORT/'RESULTS.md').exists()
 p=OUT/'presentation';rows=s['reviewed'];winner=min(VARIANTS,key=lambda v:rows[v]['window_mean']['ADE15']);recwinner=min(VARIANTS,key=lambda v:rows[v]['recording_equal']['ADE15'])
 paths=read(OUT/'source_hashes.json');changed=[k for k,h in paths.items() if sha(ROOT/k)!=h];assert not changed
 lines=['# 第二轮扩充预测结果观看入口','',
 '[先看本批固定10例六组同步视频](fixed_second10_six_groups.mp4)（42秒，半速）。白圈为 SAM 参考中心，橙圈为预测中心。','',
 '- [原始固定9例](fixed_original9_six_groups.mp4)',
 '- [第一批扩充固定9例](fixed_first9_six_groups.mp4)',
 '- [本批事后改善与退化案例](selected_second_gain_loss_six_groups.mp4)：事后选择，不能代表平均表现。',
 '- [完整报告](../../../reports/transformer_expanded_v2_20261005/RESULTS.md)','',
 '主数据共296窗、13个录制、三个种子。旧199窗的测试标签、验证窗口、相机权重和训练预算固定，新增允许的训练窗口。',
 f'六组窗口平均 ADE 最低为 {NAMES[winner]}：{rows[winner]["window_mean"]["ADE15"]:.2f}px；录制等权最低为 {NAMES[recwinner]}。保持最后位置 ADE 为 {rows["hold_last"]["window_mean"]["ADE15"]:.2f}px。','',
 '未来帧仅用于展示和评分。旧结果与原选定基线均已保留。']
 (p/'START_HERE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 values=', '.join(f'{v[0]} {rows[v]["window_mean"]["ADE15"]:.2f}/{rows[v]["window_mean"]["FDE15"]:.2f}' for v in VARIANTS)
 delta=', '.join(f'{v[0]} {s["old_test_before_after"][v]["delta_ADE"]:+.2f}' for v in VARIANTS)
 record=f'''Completed on {datetime.now().date()}: 234 trajectory models, 296 windows,
13 recording folds, three seeds, 33 exact reused cameras and six new held-out
cameras. ADE/FDE px: {values}.
Hold-last {rows['hold_last']['window_mean']['ADE15']:.2f}/{rows['hold_last']['window_mean']['FDE15']:.2f}.
Lowest learned window ADE: {winner}; recording-equal: {recwinner}.
Matched old199 test ADE changes: {delta}. Previous eleven validation window
indices, test labels and camera weights were fixed; new validation-recording
windows withheld unused in those folds. Added97 and old199 evaluated separately;
13 marked static controls and sparse/recording-equal diagnostics retained.
All234 checkpoint replays/702 prediction files, future isolation, independent
geometry, 198 previous-model runtime replays and four full video decodes passed.
Protected {len(paths)} old files unchanged. No test-driven tuning or selected
baseline replacement. See `reports/transformer_expanded_v2_20261005/RESULTS.md`.

'''
 ag=ROOT/'AGENTS.md';text=ag.read_text(encoding='utf8');marker='## Approved second G1 data expansion'
 assert marker in text and 'Completed on '+str(datetime.now().date())+': 234 trajectory models' not in text
 ag.write_text(text.replace(marker,record+marker,1),encoding='utf8')
 files=list((ROOT/'scripts'/OUT.name).glob('*.py'))+list(REPORT.glob('*.md'))+[f for f in OUT.glob('*.json') if f.name not in ['delivery_manifest.json','DELIVERED.json']]+list(DATA.glob('*'))+[p/'START_HERE.md']
 write(OUT/'delivery_manifest.json',dict(version=OUT.name,completed=str(datetime.now()),samples=296,models=234,reused_cameras=33,new_cameras=6,protected_files=len(paths),changed_sources=changed,files={f.relative_to(ROOT).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files if f.is_file()},videos=media['checks']))
 write(OUT/'DELIVERED.json',dict(passed=True,report=str((REPORT/'RESULTS.md').relative_to(ROOT)),review_entry=str((p/'START_HERE.md').relative_to(ROOT)),old_assets_preserved=True))
 print('Result, media index and persistent memory saved.',flush=True)
if __name__=='__main__':main()
