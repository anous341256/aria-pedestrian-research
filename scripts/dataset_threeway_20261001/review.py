"""Persist the assistant's explicit visual observations, not automatic GT labels."""
from audit import ROOT,OUT,read,save

RECORDING_NOTES={
'150809':('G2','相机转向突出；多数人物候车，局部有人近身通过、柱子与售货机遮挡。110–115秒可作相对简单的远人/静止对照。'),
'151153':('G3','站台下车客流、同向跟随、扶梯/楼梯与站厅交叉；光照和楼层同时变化。120–125秒虽较稳，却缺少持续可用的行人。'),
'151841':('G3','站台进入车厢；近距离人物、车门和座椅遮挡，转头与室内外切换并存。'),
'155313':('G3','站厅闸机及商业入口高密度双向客流，多次近身遮挡、转头与曝光变化。'),
'162338':('G3','短片中迎面多人；头发、手和手机进入视野；缺MPS SLAM，运动强度暂不能量化。'),
'162538':('G3','食品卖场内跟随、迎面穿行、停留同时存在，柜台/人体遮挡与头部转动叠加。'),
'163303':('G3','室内通道到室外出口再下扶梯；双向客流、光照变化、扶手与手入镜。'),
'163506':('G3','楼梯和相邻扶梯同时有多人；俯仰、空间层次及栏杆遮挡混合。'),
'163528':('G3','楼梯—站厅—站台—车厢多场景；中段稀疏，后段拥挤，整段标签掩盖局部差异。'),
'164535':('G3','前方两人跟随，但扶梯口、迎面路人、手/手机、转头与近距离截断并存。'),
'164557':('G3','道路与商场转换，同行家庭/迎面路人、转头、柱子、广告人像；180–185秒稳定段在看导览牌。'),
'165112':('G3','整体有迎面行人、扶梯、模特和反射；90–100秒可单独划作相机运动突出类候选。'),
'170022':('G3','店内多人、走廊稀疏/无人、人像与橱窗反射交替；不能用低检测人数直接判简单。'),
'180234':('G3','弯折楼梯前段少人/无人，后段进入密集商场人群；转弯与近身交错混合。'),
'180512':('G3','中庭多人、强逆光、随后进入餐饮层；人数、光照、通道结构同时改变。'),
'190127':('G3','扶梯、楼层走廊与迎面人物；玻璃反射、柱子遮挡和头部转动并存，后段可继续找稀疏目标。'),
'192702':('G3','部分路段行人较少，但低照度、树/灯杆、连续迎面擦身、招牌和室内外变化仍混合。'),
'193311':('G3','前段人少且很暗、目标远；后段行人车辆和路口混合，低人数不能代表可见性良好。'),
'193524':('G3','路口聚集人群，近处人物重叠，转头模糊与低照度同时出现。'),
'193709':('G3','过街、同向与对向客流，夜间招牌亮暗反差和转头；部分目标可连续观察但非单因素场景。'),
'193956':('G3','夜间路口过街及迎面客流；近身遮挡、俯仰、侧转、明暗变化叠加。'),
'194427':('G3','拥挤车厢严重近身遮挡与截断；偏分割/身份压力测试，非典型自由步行预测。'),
'194630':('G3','约8秒短片，站台下车客流、近身交错、手机遮挡；姿态有效覆盖不足。'),
'194648':('G3','多级楼梯/扶梯、前人遮挡、站厅客流与商业通道；必须分段。'),
'201129':('G3','当前五人试验来源；同时存在转头、多人交叉、夜间亮暗反差、柱子及后续站厅转换。'),
'201425':('G3','闸机瓶颈、停走/排队、人体互遮、设备遮住下身、手入镜。'),
'201451':('G3','前后客流密集；40–50秒和55–60秒旋转小，但在扶梯上低头看手机、前人严重截断。'),
'201745':('G3','上下楼/进出车门、急转头、近身人物与手机屏幕人脸干扰。'),
'205416':('G3','站台—楼梯—闸机—出口，前人跟随和汇流，俯仰、遮挡及手/手机入镜。')}

# class, suitability, observation, remaining confounds
SEGMENT_NOTES={
'164557_180_185':(None,'exclude_forecasting','镜头稳定，但主要在看导览牌；无适合本任务的持续可见行人。','可作静态背景对照，不能补足Easy行人组。'),
'201451_040_050':('G3','visibility_diagnostic','低头看手机，前人只剩背包/下半身，扶梯运行。','相机旋转低但可见性差；不是自由步行。'),
'201451_055_060':('G3','visibility_diagnostic','前人近身且上身截断，对向扶梯出现行人。','旋转稳定，不代表身份/几何观察简单。'),
'151153_120_125':(None,'exclude_forecasting','开头边缘人物很快出画，其余主要为空站厅。','不宜将无人稳定段算成简单行人样本。'),
'150809_035_040':('G2','candidate','孤立的深色衣服候车者较清楚；镜头移动接近并经过该人，末段出画。','中等旋转加相机平移/尺度变化；不是纯旋转隔离。'),
'150809_110_115':('G1','borderline_candidate','站台画面相对平稳，红衣候车者与其他人保持空间间隔，交互较弱。','有远处多人；目标较小且多静止。仅宽松稳定阈值通过，不能代表近处自由步行。'),
'165112_090_100':('G2','candidate','一名背包行人持续在前方行走，镜头有明显转向；其他人未持续遮住该主要目标。','存在商店玻璃、人像和末段其他路人；低干扰针对主要目标，不表示全画面无人。'),
'165112_155_165':('G3','candidate','前半迎面人群与跟随目标交错，后半转头和模糊增强。','后半可另切较单纯窗口，目前整段按混合处理。'),
'170022_090_100':('G3','negative_control','最初有人近身通过，随后多为无人走廊和橱窗模特。','不推荐作为连续行人预测样本；适合作人像误检对照。'),
'170022_120_130':('G2','conditional_candidate','大部分为稀疏走廊，镜头反复转向，远处有少量行人。','目标小、玻璃内人像/反射、末段人员增多；不列首选。'),
'150809_080_090':('G3','candidate','候车者、售货机旁人物和经过的两人，视角转动并穿过柱子附近。','静止人物多，但多人/物体遮挡并不简单。'),
'192702_165_175':('G3','candidate','低照度人行道，连续迎面行人近身经过，树干/灯杆分割画面。','相机运动、弱光和靠近/出画同时发生。'),
'164557_110_120':('G2','background_diagnostic','走廊人少且镜头明显转向；近人很快从边缘离开，之后主要是远人和广告。','不列首选行人预测样本；可用于背景补偿检查。'),
'151153_020_030':('G3','candidate','起初下车客流汇入，之后相似白衣的两人并行；转头并靠近柱子。','可另选中段研究跟随，但整段有汇流、目标相似和遮挡风险。'),
'162538_010_020':('G3','candidate','卖场内连续转头、跟随与迎面穿行，近处人物和柜台共同遮挡。','人体交互、环境遮挡、目标尺度变化。'),
'193709_035_045':('G3','candidate','夜间路口过街，对向/同向人流和明显头部转动同时存在。','低照度、亮屏、行人接近和侧向交错。'),
'201129_015_025':('G3','candidate','当前五人片段中，多个目标在前方，20秒附近大幅转头使部分目标离开视野，再出现其他行人。','旋转、夜间、视野退出和重识别混合；五个已标ID不等于场景只有五个人。'),
'155313_075_085':('G3','candidate','站厅中连续近身迎面人流及横向交叉，转头和运动模糊可见。','可作白天/明亮室内的多人混合对照。')}

def main():
    inventory=read(OUT/'motion_inventory.json'); evidence=read(OUT/'segment_evidence.json')
    assert set(RECORDING_NOTES)=={r['id'] for r in inventory}
    assert set(SEGMENT_NOTES)=={r['key'] for r in evidence}
    records=[]
    for r in inventory:
        cl,note=RECORDING_NOTES[r['id']]
        records.append({'id':r['id'],'recording':r['recording'],'primary_class':cl,'classification_level':'whole-recording provisional routing label, not uniform frame label','scene':r['scene'],'duration_s':r['prior_export_duration_s'],'observation':note,'visual_basis':'12 existing sampled frames re-inspected on 2026-10-01; additional segment sheets where listed','contact_sheet':r['contact_sheet'].replace('/mnt/e/','E:/'),'pose_coverage':r['pose_coverage'],'motion_bands_seconds':r['motion_band_seconds']})
    segments=[]
    for s in evidence:
        cl,status,obs,conf=SEGMENT_NOTES[s['key']]
        segments.append({**s,'class':cl,'suitability':status,'observation':obs,'caveats':conf,'review_basis':'assistant visual inspection of every tile in this sheet; not continuous video review or annotation GT','reviewed_sample_count':len(s['sample_frames'])})
    save(OUT/'classification.json',{'status':'provisional_visual_and_sensor_audit','classes':{'G1':'相对简单/基础对照','G2':'相机运动突出、目标交互较弱','G3':'多人交互或其他观察困难与相机运动混合'},'not_applicable':'No suitable continuous pedestrian target; not a fourth difficulty class','recordings':records,'segments':segments})
    print('Saved 29 recording labels and 18 segment decisions')

if __name__=='__main__':main()
