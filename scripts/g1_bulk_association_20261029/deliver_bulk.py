from bulk_common import *

def stream_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as fp:
  for block in iter(lambda:fp.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 assert read(OUT/'packaging_complete.json')['passed']
 review=read(OUT/'visual_review_notes.json');assert review['status']=='completed_sampled_review'
 for name in ['verification.json','source_audit.json','presentation/verification.json']:assert read(OUT/name)['passed']
 old=read(OUT/'source_hashes.json');assert all(stream_hash(ROOT/p)==s for p,s in old.items())
 report=ROOT/'reports'/OUT.name/'RESULTS.txt'
 addition='\n最终目视检查：'+review['final_review_summary']+'\n左右面板的颜色和ID分别在各自方法内编号，不能只按颜色判断跨方法对应关系。\n人工复核疑点记录在visual_review_notes.json；保留原自动质量分级，不把抽样检查冒充逐帧人工GT。\n'
 if '最终目视检查：' not in report.read_text(encoding='utf-8'):report.write_text(report.read_text(encoding='utf-8')+addition,encoding='utf-8')
 roots=[OUT,ROOT/'scripts'/OUT.name,ROOT/'reports'/OUT.name]
 ignore={'delivery_hashes.json','delivery_complete.json'}
 files=sorted(p for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ignore and p.suffix not in ['.log','.pyc'])
 hashes={str(p.relative_to(ROOT)):stream_hash(p) for p in files};write(OUT/'delivery_hashes.json',hashes)
 assert all(stream_hash(ROOT/p)==s for p,s in hashes.items())
 write(OUT/'delivery_complete.json',dict(passed=True,files=len(files),protected_files=len(old),final_visual_review='sampled, not dense human annotation',bulk_forecast_training=False))
 print('DELIVERY COMPLETE',len(files),flush=True)

if __name__=='__main__':main()
