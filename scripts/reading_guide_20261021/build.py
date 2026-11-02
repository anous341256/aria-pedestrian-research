from pathlib import Path
import json,html,hashlib,zipfile,shutil,re,sys
from prepare import OUT,PACK
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
E=html.escape
catalog=json.loads((PACK/'catalog.json').read_text(encoding='utf-8'))
# Short original study prompts. These are guides, not claimed replications.
N={
'unet':('U-Net','理解特征图如何重新变成逐像素输出。','图像 → 收缩路径提取特征 → 扩张路径恢复空间细节 → 分割图','看网络的 U 形总图；找出连接两侧特征图的跳跃连接。','为什么识别出“有人”之后，还需要恢复精细的位置？','原任务是医学图像分割；此处只借它理解编码器与解码器。'),
'yolo':('YOLO','理解一张图怎样一次输出多个目标框。','RGB 图像 → 卷积特征 → 目标框、置信分数和类别预测','先读摘要和 Unified Detection，再看检测流程与错误例子。','类别判断正确，但框的位置错了，算不算好的检测？','收录原版 YOLO 用于学习思想；后续版本的结构与训练方式并不完全相同。'),
'sam':('SAM','理解“给一个提示，分出一个对象”。','图像编码＋点／框／mask 提示编码 → mask 解码器 → 像素区域','先看模型总图，再看提示方式；暂时跳过大规模训练细节。','一个人物框里包含背景，为什么它仍能作为分割提示？','原始 SAM 主要输出区域，不自动为每个区域提供人物等语义类别。'),
'sam2':('SAM2','从单帧分割过渡到视频中的目标传播。','当前帧特征＋提示＋历史记忆 → 当前 mask → 更新记忆','重点看模型总图中的 memory attention、memory encoder 和提示帧。','前一帧分错了，为什么可能影响后续帧？','视频 mask 传播不等于对尚未观测的未来行人位置进行预测。'),
'bytetrack':('ByteTrack','理解检测框怎样连接成带身份的轨迹。','每帧检测框 → 高分框匹配 → 利用低分框补充关联 → ID 轨迹','看两阶段关联流程；把低分但真实的目标与背景误检分开理解。','为什么直接删掉所有低分框，可能让轨迹断开？','检测和关联是不同环节；检测框多不等于身份更准确。'),
'ocsort':('OC-SORT','理解短时运动预测及遮挡期间的误差积累。','检测观测 → 运动估计与观测关联 → 轨迹状态更新','看引言的失败情形，以及 observation-centric 更新思路。','遮挡后重新看到目标，应当怎样修正之前的运动估计？','此处预测服务于跟踪关联，与长期行为预测的目标和评价不同。'),
'social_lstm':('Social LSTM','理解历史位置序列与人与人交互。','每人的历史位置 → LSTM＋social pooling → 未来位置','先看社会交互示意图和 pooling，再看输入输出及评价。','只看一个人自己的历史，会遗漏哪些绕行线索？','先核查数据中的坐标表示，不把地面轨迹直接当作原始图像像素。'),
'social_gan':('Social GAN','理解同一段历史为什么可以有多个合理未来。','历史轨迹＋随机变量 → 交互特征与生成器 → 多种未来轨迹','看生成器、判别器和多样性目标；比较单条预测与多个样本。','从很多条预测中选最接近真值的一条，能否当作实际部署表现？','多候选 best-of-K 指标必须注明候选数，不能直接比较单输出结果。'),
'social_stgcnn':('Social-STGCNN','把人之间的交互表示成图。','历史位置 → 时空图 → 图卷积与时间建模 → 未来轨迹','找清楚节点、边和时间维度分别表示什么。','两个人在图像中靠近，是否一定在真实空间中靠近？','邻接关系与坐标定义共同决定交互的含义。'),
'pecnet':('PECNet','理解先猜终点，再补整条路径的预测思路。','历史轨迹 → 候选终点 → 终点条件与社会交互 → 未来路径','先看终点预测与轨迹预测的连接方式。','终点猜错时，中间轨迹能否仍然合理？','预测终点与已知真实终点是不同实验条件。'),
'agentformer':('AgentFormer','理解同时沿时间与不同人物交换信息。','多人的历史状态 → agent-aware attention → 概率未来轨迹','看 token 表示、人物身份编码和时空注意力。','模型如何分辨两个来自不同人的相似轨迹片段？','注意力权重不能直接解释为某因素的物理贡献。'),
'mid':('MID','认识用扩散过程表达未来运动不确定性。','历史条件＋噪声 → 逐步去噪 → 候选未来轨迹','先看任务定义与扩散示意图，公式可在第二轮阅读。','增加候选数量与真正改善预测分布有什么区别？','属于生成式预测方法；并不自动解决移动相机造成的观测混合。'),
'trajectron':('Trajectron++','理解道路场景中多类参与者、地图与动力学。','历史状态＋交互图＋地图等输入 → 概率模型与动力学 → 多种未来','先列出每种输入，再看模型如何约束运动可行性。','如果去掉地图或车辆动力学约束，哪些假设还成立？','本阶段是应用背景分类；该模型也涉及行人，不是只预测汽车。'),
'pedformer':('PedFormer','理解运动、姿态和场景信息怎样融合。','多种历史观测 → 跨模态编码与运动条件场景信息 → 轨迹及过街预测','看模型总图与 SAIM；逐项标记输入来源及时间范围。','加入场景特征是否一定有益？消融怎样回答这个问题？','车辆与可穿戴相机的运动条件不同；使用哪些自车信息需逐项核查。'),
'decouple':('Decouple','理解双分支为什么出现，以及分支如何受监督。','车辆相关输入＋行人相关输入 → 双塔与动作感知监督 → 预测','用两种颜色标出两个分支的数据流与损失。','网络分成两支，就足以证明真实运动完全分开了吗？','相机分支用到车辆动作、速度等信息；不是无条件的 RGB-only 系统。'),
'trajmamba':('TrajMamba','在基础模型之后了解近期时序编码方法。','行人与自车运动 → 双路 Mamba 编码 → 自运动引导解码 → 未来框','看编码与解码之间的信息传递；不必先推导 Mamba 全部公式。','与普通特征拼接相比，自运动引导具体发生在哪里？','收录本地 arXiv v1；正式发表状态以来源页面为准。'),
'yagi':('Yagi · FPL','进入“戴着相机，预测眼前的人”的任务。','人物历史位置、尺度、姿态＋自运动线索 → 多流模型 → 未来图像位置','先看任务示意图，再看 location/scale、pose、ego-motion 三类输入。','自己转头时，静止的人为什么也会在画面中移动？','它预测周围的人；不要与预测佩戴者自身路线混淆。'),
'neumann':('Neumann & Vedaldi','理解未来相机视角为什么影响最终图像预测。','历史单目观测与几何信息 → 行人与自车运动预测 → 未来视角位置','分别标记过去运动估计、未来运动预测和几何组合。','历史补偿做好之后，为什么仍然需要未来相机信息？','这是车载研究，放在这里作为从几何处理到未来视角预测的桥梁。'),
'qiu_indoor':('Qiu · Indoor FPL','看一个明确、容易画出输入输出的多模态实例。','人物框＋25 个人体关键点＋6 轴 IMU → LSTM 编解码器 → 未来框','看 Figure 1 和方法部分，手画每帧特征如何拼接。','这里的 pose 是人体骨架，还是相机位姿？','IMU 与人体姿态同时增加的对照，不能单独说明其中某一项的贡献。'),
'headsup':('HEADS-UP','理解头戴相机的数据条件与场景分层。','头戴 RGB／深度／运动信息 → 轨迹与坐标处理 → 预测与风险分析','优先看数据集部分、Easy/Hard/Uncontrolled 图和 semi-local 坐标。','控制相机运动与控制行人交互，分别隔离了什么困难？','我们 G1/G2/G3 是借鉴后的事后分类，并非完全复刻其采集协议。'),
'biflow':('BiFlow / EgoTraj-Bench','理解预测前的观测可能已经不准确。','带噪历史轨迹 → 历史去噪／重建与未来预测 → 未来轨迹','先看噪声类型和模型总图，再看历史重建的作用。','如果训练历史很干净、部署时有 ID 切换，会发生什么？','原研究对齐第一人称噪声观测与 BEV 参考；不等于我们的像素坐标实现。'),
'nim_stgcn':('NIM-STGCN','学习深度、相机位姿和内参如何串成坐标变换。','历史图像／人物框 → 深度与位姿估计 → 统一视角 → 补全与预测','先画数据流；标明每次变换的输入坐标系、输出坐标系及所需量。','统一视角后，平移、视差及估计噪声还会留下多少？','先理解方法；评价坐标、误差单位和候选选择需要另行核对。'),
'depth_fusion':('距离特征融合','学习把深度压缩成每人每帧的数值。','人物框＋单目深度图 → 区域距离序列 → 动态特征融合 → 过街分类','读 Relative Distance 小节与整体融合图。','有了人物框大小，距离信息还可能补充什么？','目标是过街意图分类；单目相对深度不等于双目米制深度。'),
'vptp':('V-PTP-IC','了解深度统计量与其他特征的联合编码。','轨迹、深度、场景、交互 → 统一特征与注意力 → LSTM 预测','重点看 Unified Feature Processing Module，列出各编码器输入。','深度数值与整张场景深度图的作用有什么不同？','作为结构参考；不要在坐标、尺度及评价协议未对齐时照搬性能数字。'),
'qiu_multimodal':('Qiu · 多模态融合','理解整张深度图作为环境上下文的方案。','自身历史＋附近人历史＋语义／深度图编码 → 级联 cross-attention → 自身未来','先确认预测对象，再看各模态编码与 cascaded cross-attention。','与给每个人一个距离值相比，整张深度图多保留了什么？','这篇预测佩戴者自身；与 Qiu 2021 预测周围人的论文不同。'),
'foundation_stereo':('FoundationStereo','弄清楚深度特征从哪里来。','经过校正的左右图像 → 立体匹配 → 视差 → 结合标定换算深度','先看双目输入、视差输出，再了解跨域泛化；把标定与网络分开。','为什么得到一张看起来平滑的深度图，不等于测距一定准确？','模型主要解决双目匹配；RGB 对齐、mask 统计和质量检查仍需另外完成。'),
'grud':('GRU-D','学习怎样告诉模型“这个数没有测到”。','时序值＋有效标记＋时间间隔 → 衰减与 GRU → 预测','看 missingness 的 mask/time interval 定义与模型示意图。','缺失值填零与真实测量为零，怎样区分？','原实验属于通用／临床时序任务；收录作者预印本，期刊发表年份为 2018。'),
}
ST=[
(1,'从像素到特征','先理解图像怎样进入网络','图像是数值数组，特征是网络计算得到的表示。先理解输入、特征与输出，不急着背模型名字。','能画出“图像 → 特征 → 输出”的流程，并解释训练与推理的区别。'),
(2,'检测与分割','找到人，再分出这个人','用 YOLO、SAM 和 SAM2 串起目标框、像素区域以及视频传播。','能解释检测框与 mask 的区别，以及提示式分割需要什么输入。'),
(3,'视频跟踪','跨帧认出同一个人','检测回答“这一帧哪里有人”，跟踪还需要回答“下一帧是不是同一个人”。','能解释一次漏检、一次 ID 切换为何会影响轨迹数据。'),
(4,'固定视角下的轨迹预测','从历史位置到多种未来','先看地面／鸟瞰表示中的行人交互，再学习概率预测。鸟瞰是一种表示，不一定来自头顶摄像机。','能区分历史窗口、预测窗口、ADE、FDE，以及单输出与 best-of-K。'),
(5,'自动驾驶中的预测','把自车、道路和其他参与者加进来','移动观察者带来自运动；道路、地图与车辆动力学也成为额外信息。','能列出一个模型实际使用的自车、地图和行人信息。'),
(6,'可穿戴第一人称预测','观察者也在走路和转头','Ego 不只指 AR：车载、胸前与头戴相机都可以是第一人称。这里聚焦头戴与可穿戴场景。','能分清“预测周围的人”和“预测佩戴者自己”，并说明相机旋转的作用。'),
(7,'多模态与深度特征','让距离、图像和运动提供补充','把辅助信息的来源、时间范围、质量和融合位置说清楚，再讨论复杂结构。','能提出一个有／无深度的公平对照，并正确处理缺测。'),
]
BRIDGES={
1:'''<div class="concept-grid"><div><b>01 · 像素</b><p>一张彩色图可以写成 H × W × 3 个数。三个通道记录红、绿、蓝；像素坐标表示它在图中的位置。</p></div><div><b>02 · 特征</b><p>卷积用可学习的局部滤波器生成特征图。较深的网络组合更多信息，形成适合任务的表示；特征不是预先写好的人体规则。</p></div><div><b>03 · 训练与推理</b><p>训练时用样本和目标计算损失、更新参数；推理时用学到的参数处理新输入。测试资料不能用来选择训练方案。</p></div><div><b>04 · 注意力</b><p>Transformer 把输入组织成 token，通过注意力交换信息。读 SAM 时，先区分图像特征与提示特征即可，不必立刻推导全部公式。</p></div></div><p class="mini">本页是原创入门说明。可联网拓展：<a href="https://cs231n.github.io/convolutional-networks/" target="_blank" rel="noopener">CS231n 卷积网络教程 ↗</a>。离线学习不依赖该链接。</p>''',
2:'''<div class="concept-grid"><div><b>检测 · Object detection</b><p>输出人物框、类别和分数。框告诉我们大致位置，但框内也可能有背景。</p></div><div><b>分割 · Segmentation</b><p>输出像素区域。语义分割区分类别；实例分割进一步区分同类中的不同个体。</p></div><div><b>提示式分割 · Promptable</b><p>给 SAM 一个点、框或 mask，让它分出指定对象。区域本身不自动等同于带类别的检测结果。</p></div><div><b>视频传播 · Propagation</b><p>SAM2 利用历史记忆传播 mask。仍需关注身份、遮挡与错误累积，不能把自动输出直接当成人工真值。</p></div></div>''',
3:'''<div class="bridge"><b>最常见的三种问题</b><p>漏检：这一帧没找到人。轨迹中断：同一个人的记录断成两段。ID 切换：把另一个人的位置接到了原轨迹里。后两者会把错误的运动关系送给预测器。</p></div>''',
4:'''<div class="concept-grid"><div><b>过去与未来</b><p>历史窗口是允许观察的片段；预测窗口是尚未观察、需要预测的片段。例如使用过去 1.5 秒，预测未来 0.5 秒。</p></div><div><b>ADE 与 FDE</b><p>ADE 是各未来时间点位置误差的平均；FDE 是最后一个预测时间点的误差。先确认坐标单位是像素还是米。</p></div><div><b>先看简单基线</b><p>保持最后位置、匀速外推都值得比较。复杂模型只有超过合理基线，才说明增加的复杂度有实际价值。</p></div><div><b>一个未来与多个未来</b><p>人可以左绕，也可以右绕。best-of-K 用真值从 K 条候选里挑最接近的一条；它与实际选择一条输出的成绩不同。</p></div></div>''',
5:'''<div class="bridge"><b>几何桥梁：画面里的移动 ≠ 只有目标自己在移动</b><p>像素坐标是“画面上的位置”；相机坐标是“以镜头为原点描述方向与位置”；世界坐标是选定的外部参考。内参把像素与视线方向联系起来，位姿描述相机的位置和朝向。只有视线方向还不知道人有多远，深度提供这一维信息。</p><p>旋转补偿把历史观测统一到一个朝向；未来重投影把预测位置放回未来相机画面。处理相机平移时，还必须考虑目标深度和视差。</p></div>''',
6:'''<div class="bridge"><b>读之前先写一句任务定义</b><p>“根据过去的行人运动和相机运动，预测周围行人在未来画面中的位置。”若论文预测的是佩戴者自己在世界中的路线，就标为邻近任务。人体 pose（骨架）与 camera pose（位姿）也要分开。</p></div>''',
7:'''<div class="bridge"><b>两种深度输入，先不要混在一起</b><p>目标级：每个人每帧一个代表深度，再附上变化量、有效标记和质量线索。场景级：把整张深度图编码成特征，保留环境空间结构。它们回答的问题不同。</p><p>深度变化同时包含观察者、目标和测量的影响，不直接等于真实行人速度。只输入历史测量；缺测不能伪装成真实的零米。我们的入门目标是理解特征输入，不要求此时完成三维几何预测。</p></div>'''
}
assert set(N)=={c['id'] for c in catalog}
for c in catalog:
 short,why,flow,read,q,limit=N[c['id']]
 c.update(short=short,why=why,flow=flow,read=read,question=q,limit=limit)

def card(c):
 search=' '.join(str(c[k]) for k in ('short','title','year','why','flow','read','question','limit'))
 return f'''<article class="paper" id="paper-{c['id']}" data-search="{E(search)}" data-level="{c['level']}">
 <div class="paper-top"><span class="tag {'extra-tag' if c['level']=='extra' else ''}">{'选读' if c['level']=='extra' else '主读'}</span><span class="year">{E(c['year'])}</span><label class="read-check"><input type="checkbox" data-read="{c['id']}"> 已读</label></div>
 <h3>{E(c['short'])}</h3><p class="full-title" lang="en">{E(c['title'])}</p><p class="why">{E(c['why'])}</p>
 <div class="flow">{E(c['flow'])}</div>
 <dl><dt>先看哪里</dt><dd>{E(c['read'])}</dd><dt>读后回答</dt><dd>{E(c['question'])}</dd></dl>
 <details class="boundary"><summary>理解边界</summary><p>{E(c['limit'])}</p></details>
 <div class="paper-actions"><a class="pdf-link" href="{c['file']}" target="_blank" rel="noopener" aria-label="打开本地 PDF：{E(c['short'])}">打开本地 PDF <span aria-hidden="true">↗</span></a><span class="file-info">{c['pages']} 页 · {c['bytes']/1048576:.1f} MB</span><a class="source-link" href="{c['ref']}" target="_blank" rel="noopener">来源（联网）</a></div></article>'''
sections=[]
for n,title,sub,intro,outcome in ST:
 main=[c for c in catalog if c['stage']==n and c['level']=='main'];extra=[c for c in catalog if c['stage']==n and c['level']=='extra']
 extra_html=f'<details class="extras"><summary>选读拓展 · {len(extra)} 篇 <span>完成本阶段后再看</span></summary><div class="cards">'+''.join(map(card,extra))+'</div></details>' if extra else ''
 sections.append(f'''<section class="stage" id="stage-{n}"><div class="section-title"><span class="step">{n:02d}</span><div><p class="eyebrow">{sub}</p><h2>{title}</h2></div></div><p class="stage-intro">{intro}</p>{BRIDGES[n]}<div class="outcome"><b>这一阶段学会：</b>{outcome}</div><div class="cards">{''.join(map(card,main))}</div>{extra_html}</section>''')
nav=''.join(f'<a href="#stage-{n}" data-nav="{n}"><span>{n:02d}</span>{t}</a>' for n,t,*_ in ST)
rows=''.join(f'<tr><td>{c["stage"]:02d}</td><td>{"主读" if c["level"]=="main" else "选读"}</td><td><a href="{c["file"]}" target="_blank" rel="noopener">{E(c["short"])}</a><small>{E(c["title"])}</small></td><td>{E(c["year"])}</td><td>{c["pages"]}</td></tr>' for c in catalog)
main_count=sum(c['level']=='main' for c in catalog)
template=Path(__file__).with_name('template.html').read_text(encoding='utf-8')
for key,val in {'NAV':nav,'SECTIONS':''.join(sections),'ROWS':rows,'COUNT':str(len(catalog)),'MAINCOUNT':str(main_count),'SIZE':f'{sum(c["bytes"] for c in catalog)/1048576:.0f}','DATA':json.dumps([{'id':c['id'],'title':c['title'],'stage':c['stage'],'level':c['level']} for c in catalog],ensure_ascii=False).replace('</',r'<\/')}.items():
 template=template.replace('{{'+key+'}}',val)
(PACK/'index.html').write_text(template,encoding='utf-8')
(PACK/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
readme='''计算机视觉到第一人称轨迹预测 · 离线导读包
版本：2026-10-21

使用方法
1. 先把整个 ZIP 解压到任意文件夹。
2. 双击 index.html，用 Edge、Chrome、Firefox 等浏览器打开。
3. 点击“打开本地 PDF”即可阅读。请勿只发送 index.html，papers 文件夹必须随包保留。
不需要安装 Python、启动服务器、注册账号或联网。
浏览器若下载 PDF 而不是预览，可用系统 PDF 阅读器打开。

本包包含 27 篇 PDF、中文导读、完整书目表、阅读记录模板与校验清单。
页面默认展示主读，选读在各阶段下方展开。搜索会同时检索选读。
阅读勾选和笔记尝试保存于当前浏览器；本地文件权限、移动路径或清理数据可能使其失效。
需要保留或转移记录时，点击“导出阅读记录”。导出的 JSON 可以重新导入。

资料来源
原论文来自已有本地文献与作者/会议/出版社公开版本。
每份 PDF 按原文件保留，未裁剪、改写或删除署名；个别收录版本为预印本。
论文版权与许可归原作者/出版方；随组内学习资料传递时请保留署名与来源，
公开再次分发前应逐项核查原文许可。本导读的概念说明和阅读问题为整理者撰写，
不是论文作者原话，也不代表所有方法已复现。

隐私与离线
页面不使用 CDN、第三方字体、统计脚本或远程图片；打开页面不会自动请求网络。
外部“来源”链接和教程链接仅点击时联网。
包内未包含个人课程资料、采集视频、训练数据或用户账户路径。

文件
index.html             导读首页
papers/                全部论文 PDF（英文短文件名，便于转发）
catalog.json           完整书目、导读与 PDF 哈希
论文阅读清单.txt       不依赖网页的纯文本清单
阅读记录模板.txt       可直接填写的阅读模板
checksums.sha256       文件 SHA-256 校验值
'''
(PACK/'README_先读我.txt').write_text(readme,encoding='utf-8-sig')
lines=['论文阅读清单｜2026-10-21','全部论文在本包 papers 文件夹，可离线阅读。','']
for n,title,*_ in ST:
 lines += [f'{n:02d} {title}']
 for c in catalog:
  if c['stage']==n: lines += [f'  [{"主读" if c["level"]=="main" else "选读"}] {c["short"]} | {c["year"]}',f'  {c["title"]}',f'  本地：{c["file"]}',f'  来源：{c["ref"]}',f'  阅读目的：{c["why"]}','']
(PACK/'论文阅读清单.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
(PACK/'阅读记录模板.txt').write_text('论文：\n阅读人／日期：\n\n1. 要解决什么问题？预测谁？\n2. 输入是什么？哪些时间点允许输入？\n3. 输出是什么？在哪个坐标系、用什么单位？\n4. 一句话／一张图说明数据流：\n5. 哪个实验最支持论文观点？\n6. 一个局限或还没看懂的问题：\n7. 和我们研究的联系：\n',encoding='utf-8-sig')
print(f'Built {len(catalog)} papers / {main_count} main, {PACK}')
