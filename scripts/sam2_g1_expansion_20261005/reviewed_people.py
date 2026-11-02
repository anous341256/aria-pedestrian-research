"""Visual census decisions from full-frame sheets and temporal crop strips.

Detector aliases are explicitly bounded at observed switches. These are local
annotation identities, not globally unique people or automated associations.
"""
PEOPLE={
 '11':[('白T米色长裤，同向',[1]),('白长袖蓝裤黑斜包，同向',[2]),('浅上衣蓝裤黑背包，同向',[3]),
       ('白衬衫口罩迎面，短时局部可见',[87,100,121]),('浅上衣深裤马尾，同向',[130])],
 '12':[('灰开衫白裤棕包，横向经过',[(1,0,14)]),('深蓝裙迎面，末尾局部可见',[3,(1,15,15)]),
       ('黑上衣棕裤，同向',[31]),('深衣戴帽手提袋迎面',[(40,11,31)]),('黑上衣灰裤大肩包，同向',[52]),
       ('白衬衫黑裤背包看手机',[44,(78,32,33)]),('浅上衣深裤柱旁迎面',[68])],
 '13':[('白背心黑裤，同向；与旧01/03同一人物段',[1]),('戴帽深色条纹短裤，同向',[2,72]),
       ('浅上衣深裤斜肩包，同向',[8]),('右侧浅T短裤迎面',[25]),('深蓝上衣灰裤黄色袋，迎面',[66]),
       ('右侧浅上衣深裙手提袋，迎面',[71])],
 '14':[('黑上衣浅裤Nike袋，同向；与旧09同一人物段',[1,111,160]),('蓝上衣黑短裤，同向',[2,58]),
       ('黑上衣黑裤白斜肩带，同向',[3,22,66]),('右侧浅上衣黑短裙，站立',[4,5,21]),
       ('浅花纹上衣深裙，同向',[7]),('白衬衫深马甲路边人物',[8,76]),
       ('末尾深色上衣蓝包迎面',[182]),('末尾浅上衣深裤迎面',[198])],
 '15':[('深蓝T浅短裤推行李成人，同向',[1]),('白T黑裤马尾成人，同向',[4]),
       ('深色T浅短裤迎面，短区间',[3]),('深上衣红裤迎面，短区间',[5]),
       ('近处孩子，仅可见头部的边界片段',[8,42]),('蓝T灰裤看手机人物',[10,15,19]),
       ('门前黑T短裤，同向',[29,53]),('门前白T蓝短裤，同向',[31])],
 '16':[('白T灰裤持伞马尾，同向',[1,84]),('格子上衣蓝裤马尾，同向',[6]),
       ('浅T花纹短裤背包迎面',[3]),('浅蓝上衣深裤迎面',[7]),('深色衣裤背包，同向',[5]),
       ('开头左缘花纹上衣局部人物',[2]),('白T深色背心浅裤迎面',[56])],
 '18':[('黑衣粉包马尾，同向',[(4,0,3),(7,4,49)]),('黑T灰短裤，进入车门',[2]),
       ('右侧近处白短袖眼镜人物',[6,36]),
       ('粉包人物前方白T黑背包，同向',[(11,0,6),(11,11,20),(34,17,37),(82,35,37),(93,43,49)]),
       ('栏杆后粉色开衫口罩人物',[9,15]),('栏杆后灰色上衣人物',[8,49]),
       ('栏杆后深色衬衫人物',[(10,0,16),35]),('栏杆后白短袖黑手提包人物',[45,(72,32,38)]),
       ('栏杆后白Polo深裤人物',[51,(72,31,31),(72,40,42)]),
       ('栏杆后前排白T黑背包人物',[39,53,(88,40,40)]),('栏杆后后排白衬衫人物',[(88,42,46)]),
       ('末尾栏杆后白衬衫灰裤背包人物',[89,101]),('开头远处白T深裤人物',[19])],
 '19':[('近处深蓝上衣看手机候车人物',[2]),('白衬衫深裤黑背包候车人物',[4]),('后方深上衣灰裤候车人物',[41])],
 '20':[('浅色背心米色长裤迎面',[2,(10,25,32),(16,33,35)]),
       ('深色上衣深裤迎面，遮挡后分段',[(1,0,25),17,(16,36,36)])],
 '21':[('灰蓝横条纹Polo深裤迎面',[3]),('白T深裤腰间花纹衣物迎面',[4]),
       ('黑图案T灰阔腿裤迎面',[6,36]),('深蓝裙迎面',[12]),('白T黑斜肩带迎面',[52]),
       ('深蓝T黑短裤迎面',[73]),('末尾浅色上衣深裤同向',[107]),('末尾深色无袖长裙迎面',[117,126])],
}
MANUAL={}
REJECT={
 '11':{19:'train_window_reflection',94:'train_window_reflection'},
 '12':{8:'advertisement',36:'mixed_identity_then_distant',42:'distant_ambiguous_pair',48:'distant_duplicate'},
 '13':{24:'cyclist_not_pedestrian',79:'cyclist_or_rider',80:'cyclist_or_rider',113:'cone_false_positive',122:'cyclist_or_rider',144:'distant_edge_unresolved'},
 '14':{128:'motorbike_rider',130:'motorbike_rider',132:'motorbike_rider'},
 '18':{12:'glass_reflection_of_near_white_shirt',32:'train_window_reflection',59:'train_window_reflection',76:'train_window_reflection',92:'train_window_reflection',96:'wearer_phone',104:'wearer_phone'},
 '20':{6:'building_false_positive'},'21':{113:'tree_false_positive'},
}
# Annotation retained, but outside the clear-person forecast subset by default.
REVIEW_ONLY={('15',5):'head_only_border_fragment',**{('18',i):'rail_occlusion_or_short_partial_body' for i in range(5,14)}}
REJECTED_CLIPS={'17':'Rectified review shows escalator entrance and crowd. Excluded from this G1 expansion; prepared inputs and discovery retained.'}
