from pathlib import Path
import json, hashlib, shutil, urllib.request, concurrent.futures, sys, time
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT=Path('E:/New_research')
OUT=ROOT/'outputs/reading_guide_20261021'
PACK=OUT/'CV_Trajectory_Reading_Guide'
PAPERS=ROOT/'行人轨迹预测相关论文'
TRACK=Path('C:/Users/nytha/Desktop/轨迹追踪-标注问题相关论文')
PACK.mkdir(parents=True,exist_ok=True)
(PACK/'papers').mkdir(exist_ok=True)
(OUT/'audit').mkdir(exist_ok=True)
def item(id,stage,title,year,source,ref,level='main',urls=None):
 return dict(id=id,stage=stage,title=title,year=year,source=str(source) if source else None,ref=ref,level=level,urls=urls or [],file=f'papers/{id}.pdf')
C=[
item('unet',1,'U-Net: Convolutional Networks for Biomedical Image Segmentation','2015',Path('E:/都是PDF/U-NET论文.pdf'),'https://arxiv.org/abs/1505.04597','extra'),
item('yolo',2,'You Only Look Once: Unified, Real-Time Object Detection','CVPR 2016',None,'https://arxiv.org/abs/1506.02640',urls=['https://arxiv.org/pdf/1506.02640','https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf']),
item('sam',2,'Segment Anything','ICCV 2023',None,'https://arxiv.org/abs/2304.02643',urls=['https://arxiv.org/pdf/2304.02643','https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf']),
item('sam2',2,'SAM 2: Segment Anything in Images and Videos','2024 首发',None,'https://arxiv.org/abs/2408.00714',urls=['https://arxiv.org/pdf/2408.00714']),
item('bytetrack',3,'ByteTrack: Multi-Object Tracking by Associating Every Detection Box','ECCV 2022',TRACK/'136820001.pdf','https://arxiv.org/abs/2110.06864'),
item('ocsort',3,'Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking','CVPR 2023',TRACK/'Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf','https://arxiv.org/abs/2203.14360','extra'),
item('social_lstm',4,'Social LSTM: Human Trajectory Prediction in Crowded Spaces','CVPR 2016',PAPERS/'Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf','https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html'),
item('social_gan',4,'Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks','CVPR 2018',PAPERS/'1803.10892v1.pdf','https://arxiv.org/abs/1803.10892'),
item('social_stgcnn',4,'Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction','CVPR 2020',PAPERS/'2002.11927v3.pdf','https://arxiv.org/abs/2002.11927','extra'),
item('pecnet',4,'It is not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction','ECCV 2020',PAPERS/'2004.02025v3.pdf','https://arxiv.org/abs/2004.02025','extra'),
item('agentformer',4,'AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting','ICCV 2021',PAPERS/'2103.14023v3.pdf','https://arxiv.org/abs/2103.14023','extra'),
item('mid',4,'Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion','CVPR 2022',PAPERS/'2203.13777v1.pdf','https://arxiv.org/abs/2203.13777','extra'),
item('trajectron',5,'Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data','ECCV 2020',PAPERS/'2001.03093v5.pdf','https://arxiv.org/abs/2001.03093'),
item('pedformer',5,'PedFormer: Pedestrian Behavior Prediction via Cross-Modal Attention Modulation and Gated Multitask Learning','ICRA 2023',PAPERS/'2210.07886v1.pdf','https://arxiv.org/abs/2210.07886'),
item('decouple',5,'Decouple Ego-View Motions for Predicting Pedestrian Trajectory and Intention','TIP 2024',PAPERS/'10621007.pdf','https://doi.org/10.1109/TIP.2024.3445734'),
item('trajmamba',5,'TrajMamba: An Ego-Motion-Guided Mamba Model for Pedestrian Trajectory Prediction from an Egocentric Perspective','2026；所收录为 arXiv v1',PAPERS/'2603.14739v1.pdf','https://arxiv.org/abs/2603.14739','extra'),
item('yagi',6,'Future Person Localization in First-Person Videos','CVPR 2018',PAPERS/'Yagi_Future_Person_Localization_CVPR_2018_paper.pdf','https://arxiv.org/abs/1711.11217'),
item('neumann',6,'Pedestrian and Ego-Vehicle Trajectory Prediction From Monocular Camera','CVPR 2021',PAPERS/'Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf','https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html','extra'),
item('qiu_indoor',6,'Indoor Future Person Localization from an Egocentric Wearable Camera','IROS 2021',None,'https://arxiv.org/abs/2103.04019',urls=['https://arxiv.org/pdf/2103.04019']),
item('headsup',6,'HEADS-UP: Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems','2024；所收录为 arXiv v1',TRACK/'2409.20324v1.pdf','https://arxiv.org/abs/2409.20324'),
item('biflow',6,'EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations','2025 首发／2026 修订',PAPERS/'EgoTraj_Bench_BiFlow_arXiv_2510.00405v2.pdf','https://arxiv.org/abs/2510.00405','extra'),
item('nim_stgcn',6,'NIM-STGCN: Differentiable motion decomposition for egocentric pedestrian trajectory prediction','2025 在线／2026 卷期',Path('E:/s40747-025-02190-3.pdf'),'https://doi.org/10.1007/s40747-025-02190-3','extra'),
item('depth_fusion',7,'Pedestrian Cross Forecasting with Hybrid Feature Fusion','ACML 2023／论文集 2024',None,'https://proceedings.mlr.press/v222/dong24a.html',urls=['https://proceedings.mlr.press/v222/dong24a/dong24a.pdf']),
item('vptp',7,'V-PTP-IC: End-to-End Joint Modeling of Dynamic Scenes and Social Interactions for Pedestrian Trajectory Prediction from Vehicle-Mounted Cameras','Sensors 2025',Path('E:/下载/sensors-25-07151.pdf'),'https://doi.org/10.3390/s25237151','extra'),
item('qiu_multimodal',7,'Egocentric Human Trajectory Forecasting with a Wearable Camera and Multi-Modal Fusion','RA-L 2022',None,'https://arxiv.org/abs/2111.00993','extra',urls=['https://arxiv.org/pdf/2111.00993']),
item('foundation_stereo',7,'FoundationStereo: Zero-Shot Stereo Matching','CVPR 2025',None,'https://openaccess.thecvf.com/content/CVPR2025/html/Wen_FoundationStereo_Zero-Shot_Stereo_Matching_CVPR_2025_paper.html',urls=['https://openaccess.thecvf.com/content/CVPR2025/papers/Wen_FoundationStereo_Zero-Shot_Stereo_Matching_CVPR_2025_paper.pdf','https://arxiv.org/pdf/2503.09898']),
item('grud',7,'Recurrent Neural Networks for Multivariate Time Series with Missing Values','Scientific Reports 2018',None,'https://doi.org/10.1038/s41598-018-24271-9','extra',urls=['https://arxiv.org/pdf/1606.01865','https://www.nature.com/articles/s41598-018-24271-9.pdf']),
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prep(c):
 p=PACK/c['file']; origin=c['source']; acquisition='existing_local'
 if origin:
  src=Path(origin)
  if not src.is_file(): raise FileNotFoundError(src)
  shutil.copy2(src,p)
 else:
  acquisition='official_download'
  if not p.exists():
   errors=[]
   for url in c['urls']:
    try:
     req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (academic offline reading)'})
     with urllib.request.urlopen(req,timeout=75) as resp: data=resp.read()
     if not data.startswith(b'%PDF-'): raise ValueError('not PDF')
     p.write_bytes(data);origin=url;break
    except Exception as e: errors.append(str(e))
   else: raise RuntimeError(c['id']+': '+str(errors))
  else: origin=c['urls'][0]
 reader=PdfReader(p)
 extracts=[]
 for i,page in enumerate(reader.pages):
  txt=page.extract_text() or ''
  extracts.append(f'\n--- PDF PAGE {i+1} ---\n'+txt)
 (OUT/'audit'/f"{c['id']}.txt").write_text(''.join(extracts),encoding='utf-8',errors='replace')
 c.update(pages=len(reader.pages),bytes=p.stat().st_size,sha256=sha(p),acquisition=acquisition)
 if c['source']:
  assert sha(Path(c['source']))==c['sha256']
 c['acquired_from']=origin
 print(json.dumps({'id':c['id'],'pages':c['pages'],'title_excerpt':extracts[0][0:260]},ensure_ascii=False),flush=True)
 return c
if __name__=='__main__':
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(prep,c):c for c in C}
  for f in concurrent.futures.as_completed(futures): results.append(f.result())
 results.sort(key=lambda c:[x['id'] for x in C].index(c['id']))
 (OUT/'audit'/'acquisition_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 # Public catalog intentionally omits original local paths, including the user profile.
 public=[{k:v for k,v in c.items() if k not in ('source','urls','acquired_from')} for c in results]
 (PACK/'catalog.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
 print('COMPLETE',len(public),sum(c['bytes'] for c in public),flush=True)
