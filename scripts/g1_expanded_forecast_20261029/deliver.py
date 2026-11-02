from expanded_common import *

def main():
 for p in ['verification.json','independent_data_audit.json','independent_prediction_audit.json','presentation/gru/verification.json','presentation/d/verification.json','visual_review.json']:assert read(OUT/p)['passed']
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/p)==s for p,s in protected.items())
 (OUT/'presentation/START_HERE.txt').write_text('同一原117测试窗口。\nGRU：gru/fixed_9_gru_expanded.mp4。左保存原GRU，中原数据继续训练，右原数据＋新G1。\nD：d/fixed_9_d_expanded.mp4。左保存原GRU，中D仅原数据，右D原数据＋新G1。\n两段均固定原9例seed17，半速37.8秒，未按结果挑选。\n白点为SAM参考中心，橙点为预测；各目录comparison.png为汇总。\n训练新增299自动质量G1窗口，原版本保留。\n',encoding='utf-8')
 report=ROOT/'reports'/OUT.name/'RESULTS.txt'
 with report.open('a',encoding='utf-8') as fp:fp.write('\n目视检查：两组汇总图与各9例末帧均已检查，成功和失败案例保留。两段视频共756帧均完整解码验证。\n')
 paths=[p for folder in [OUT,ROOT/'scripts'/OUT.name,report.parent] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery_hashes.json','delivery_complete.json'] and p.suffix!='.log']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};write(OUT/'delivery_hashes.json',hashes);assert all(sha(ROOT/p)==s for p,s in hashes.items());write(OUT/'delivery_complete.json',dict(passed=True,files=len(hashes),protected_files=len(protected),models=72,selected_baseline_replaced=False));print('DELIVERED',len(hashes))

if __name__=='__main__':main()
