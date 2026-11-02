from pathlib import Path
import concurrent.futures, hashlib, json, re
import urllib.request, io
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path('E:/New_research/papers/ARVR_Autonomous_Driving_20261029')
PAPERS = [
 ('01','ARVR','Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span','NeurIPS 2025','https://arxiv.org/abs/2511.18470',[
 'https://papers.nips.cc/paper_files/paper/2025/file/bf649dbcf4eaaba4fc68ccaee25a7f9e-Paper-Conference.pdf','https://arxiv.org/pdf/2511.18470']),
 ('02','ARVR','Forecasting 3D Scanpaths in Egocentric Video','CVPR 2026','https://rehg.org/publication/pub60/',[
 'https://openaccess.thecvf.com/content/CVPR2026/papers/Ryan_Forecasting_3D_Scanpaths_in_Egocentric_Video_CVPR_2026_paper.pdf']),
 ('03','ARVR','GazeMotion: Gaze-guided Human Motion Forecasting','IROS 2024 author manuscript','https://zhiminghu.net/hu24_gazemotion.html',[
 'https://zhiminghu.net/hu24_gazemotion/pdf/hu24_gazemotion.pdf','https://arxiv.org/pdf/2403.09885v2']),
 ('04','ARVR','UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation','ICCV 2025','https://arxiv.org/abs/2508.01126',[
 'https://openaccess.thecvf.com/content/ICCV2025/papers/Patel_UniEgoMotion_A_Unified_Model_for_Egocentric_Motion_Reconstruction_Forecasting_and_ICCV_2025_paper.pdf','https://arxiv.org/pdf/2508.01126v2']),
 ('05','ARVR','The Eye-Head Mover Spectrum: Modelling Individual and Population Head Movement Tendencies in Virtual Reality','2026 arXiv v1','https://arxiv.org/abs/2602.06164',[
 'https://arxiv.org/pdf/2602.06164v1']),
 ('06','Autonomous_Driving','DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving','ICLR 2025','https://arxiv.org/abs/2503.07656',[
 'https://proceedings.iclr.cc/paper_files/paper/2025/file/a7afc9957f1190223763b6ea93218f98-Paper-Conference.pdf','https://arxiv.org/pdf/2503.07656v2']),
 ('07','Autonomous_Driving','SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation','ICRA 2025; arXiv v2 manuscript','https://arxiv.org/abs/2405.19620',[
 'https://arxiv.org/pdf/2405.19620v2']),
 ('08','Autonomous_Driving','S2-Track: A Simple yet Strong Approach for End-to-End 3D Multi-Object Tracking','2025 arXiv v2; v1 title was UA-Track','https://arxiv.org/abs/2406.02147v2',[
 'https://arxiv.org/pdf/2406.02147v2']),
]

def fetch(p):
    n,group,title,version,page,urls=p
    folder=ROOT/group
    folder.mkdir(parents=True,exist_ok=True)
    name=n+'_'+re.sub(r'[^A-Za-z0-9]+','_',title).strip('_')+'.pdf'
    path=folder/name
    result=dict(number=n,group=group,title=title,version=version,landing_page=page,file=str(path.relative_to(ROOT)),attempts=[])
    for url in urls:
        try:
            if path.exists():
                data=path.read_bytes()
                actual_url='existing local file'
            else:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    data=r.read()
                    actual_url=r.url
            if not data.startswith(b'%PDF-'): raise ValueError('Response is not a PDF')
            doc=PdfReader(io.BytesIO(data))
            count=len(doc.pages)
            if doc.is_encrypted or count<2: raise ValueError('Encrypted or implausibly short PDF')
            first=doc.pages[0].extract_text()
            # Parse every page to detect corrupt/truncated page structures.
            for pg in doc.pages: pg.extract_text()
            checkdir=ROOT/'verification'; checkdir.mkdir(exist_ok=True)
            with pdfium.PdfDocument(data) as renderdoc:
                renderdoc[0].render(scale=.65).to_pil().save(checkdir/(n+'_first_page.png'))
            if not path.exists(): path.write_bytes(data)
            result.update(status='downloaded',source_url=actual_url,pages=count,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),first_page_text=first[:1800])
            print(n,'OK',count,len(data),flush=True)
            return result
        except Exception as e:
            result['attempts'].append(dict(url=url,error=str(e)))
    result['status']='failed'
    print(n,'FAILED',result['attempts'],flush=True)
    return result

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,PAPERS))
    (ROOT/'download_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['AR/VR 与自动驾驶方向探索：论文原稿','整理日期：2026-10-29','原始 PDF 保持下载内容，不做改写或重新排版。','建议阅读顺序：01 → 06 → 08 → 02；其余作为补充。','']
    for r in results:
        lines.extend([r['number']+' '+r['title'],'版本：'+r['version'],'状态：'+r['status'],'文件：'+r['file'],'页面：'+r['landing_page'],'下载来源：'+r.get('source_url','尚未成功'),''])
    (ROOT/'00_论文目录与来源.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
