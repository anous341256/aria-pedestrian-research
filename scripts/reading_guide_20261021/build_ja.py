from pathlib import Path
import json, html, shutil, hashlib, re, sys
from ja_content import N,ST,BRIDGES
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=Path('E:/New_research/outputs/reading_guide_20261021')
SOURCE=BASE/'CV_Trajectory_Reading_Guide'
PACK=BASE/'CV_Trajectory_Reading_Guide_JA'
AUDIT=BASE/'audit_ja';AUDIT.mkdir(exist_ok=True)
PACK.mkdir(exist_ok=True)
original_hashes={p.relative_to(SOURCE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}
(AUDIT/'original_zh_hashes.json').write_text(json.dumps(original_hashes,indent=2),encoding='utf-8')
shutil.copytree(SOURCE/'papers',PACK/'papers',dirs_exist_ok=True)
catalog=json.loads((SOURCE/'catalog.json').read_text(encoding='utf-8'))
YEARS={'2024 首发':'2024 年初公開','2025 首发 / 2026 修订版':'2025 年初公開 / 2026 年改訂版','2025 在线 / 2026 卷期':'2025 年オンライン公開 / 2026 年巻号','2026；所收录为 arXiv v1':'2026 年／arXiv v1 を収録','2024；所收录为 arXiv v1':'2024 年／arXiv v1 を収録','ACML 2023／论文集 2024':'ACML 2023／論文集 2024'}
for c in catalog:
 c.update(zip(('short','why','flow','read','question','limit'),N[c['id']]))
 c['year']=YEARS.get(c['year'],c['year']).replace('首发','初公開').replace('修订','改訂').replace('在线','オンライン公開').replace('卷期','巻号').replace('预印本','プレプリント').replace('本地','収録')
E=html.escape
def card(c):
 search=' '.join(str(c[k]) for k in ('short','title','year','why','flow','read','question','limit'))
 return f'''<article class="paper" id="paper-{c['id']}" data-search="{E(search)}" data-level="{c['level']}">
 <div class="paper-top"><span class="tag {'extra-tag' if c['level']=='extra' else ''}">{'発展' if c['level']=='extra' else '基本'}</span><span class="year">{E(c['year'])}</span><label class="read-check"><input type="checkbox" data-read="{c['id']}"> 読了</label></div>
 <h3>{E(c['short'])}</h3><p class="full-title" lang="en">{E(c['title'])}</p><p class="why">{E(c['why'])}</p><div class="flow">{E(c['flow'])}</div>
 <dl><dt>読む箇所</dt><dd>{E(c['read'])}</dd><dt>考える問い</dt><dd>{E(c['question'])}</dd></dl>
 <details class="boundary"><summary>解釈上の注意</summary><p>{E(c['limit'])}</p></details>
 <div class="paper-actions"><a class="pdf-link" href="{c['file']}" target="_blank" rel="noopener" aria-label="ローカル PDF を開く：{E(c['short'])}">PDF を開く <span aria-hidden="true">↗</span></a><span class="file-info">{c['pages']} ページ · {c['bytes']/1048576:.1f} MB</span><a class="source-link" href="{c['ref']}" target="_blank" rel="noopener">出典（オンライン）</a></div></article>'''
sections=[]
for n,title,sub,intro,outcome in ST:
 main=[c for c in catalog if c['stage']==n and c['level']=='main'];extra=[c for c in catalog if c['stage']==n and c['level']=='extra']
 ex=f'<details class="extras"><summary>発展文献 · {len(extra)} 本 <span>基本を学んだ後に読む</span></summary><div class="cards">'+''.join(map(card,extra))+'</div></details>' if extra else ''
 sections.append(f'''<section class="stage" id="stage-{n}"><div class="section-title"><span class="step">{n:02d}</span><div><p class="eyebrow">{sub}</p><h2>{title}</h2></div></div><p class="stage-intro">{intro}</p>{BRIDGES[n]}<div class="outcome"><b>到達目標：</b>{outcome}</div><div class="cards">{''.join(map(card,main))}</div>{ex}</section>''')
nav=''.join(f'<a href="#stage-{n}" data-nav="{n}"><span>{n:02d}</span>{t}</a>' for n,t,*_ in ST)
rows=''.join(f'<tr><td>{c["stage"]:02d}</td><td>{"基本" if c["level"]=="main" else "発展"}</td><td><a href="{c["file"]}" target="_blank" rel="noopener">{E(c["short"])}</a><small>{E(c["title"])}</small></td><td>{E(c["year"])}</td><td>{c["pages"]}</td></tr>' for c in catalog)
T={
'lang="zh-CN"':'lang="ja"',
'从图像到未来 · 轨迹预测阅读导图':'画像から未来へ · 軌跡予測の論文ガイド',
'"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif':'"Yu Gothic UI","Meiryo","Hiragino Kaku Gothic ProN",sans-serif',
'跳转到导读正文':'本文へ移動',
'视觉 · 运动 · 预测':'視覚・運動・予測',
'七阶段学习路线':'7 段階の学習ルート',
'阅读进度':'読書の進捗',
'按自己的节奏读。<br>先看主线，再展开选读。':'自分のペースで。<br>基本から読み、発展文献へ。',
'完整论文清单':'全論文リスト',
'阅读记录 ↓':'読書メモ ↓',
'组内入门导读 / 2026.10':'研究グループ向け入門 / 日本語版',
'搜索论文、概念或关键词':'論文・概念・キーワードを検索',
'搜索论文或概念：YOLO、深度、相机…':'論文や概念を検索：YOLO、深度、カメラ…',
'>清除<':'>クリア<',
'从一张图像，<br>走向<span>未来轨迹。</span>':'1 枚の画像から、<br><span>未来の軌跡</span>へ。',
'从像素与视觉特征开始，经由检测、分割和跟踪，逐步理解行人轨迹预测，最后进入自动驾驶与可穿戴第一人称场景。':'画素と視覚特徴から始め、検出・セグメンテーション・追跡を経て歩行者軌跡予測を学び、自動運転や装着型カメラの一人称視点へ進む。',
'全部 PDF 已附':'全論文の PDF を同梱',
'离线可用':'オフライン対応',
'无需安装':'インストール不要',
'开始第一阶段 →':'ステップ 1 から始める →',
'查看 {{COUNT}} 篇论文清单':'全 {{COUNT}} 本の論文を見る',
'从图像中的人物到历史轨迹，再到多个可能未来的示意图':'画像内の人物から過去の軌跡、複数の未来候補への模式図',
'阶段 · 从基础到研究场景':'段階 · 基礎から研究課題へ',
'篇论文 · {{MAINCOUNT}} 篇主读，其余选读':'本の論文 · 基本文献 {{MAINCOUNT}} 本＋発展文献',
'本地 PDF · 约 {{SIZE}} MB 文献':'ローカル PDF · 約 {{SIZE}} MB',
'先理解问题':'まず課題を理解する',
'预测谁？观察什么？输出在哪个坐标系？先回答这些，再看网络层数。':'誰を予測するのか。何を観測するのか。出力はどの座標系か。層の数を見る前に確認する。',
'再画数据流':'次にデータの流れを描く',
'输入 → 表示与融合 → 输出。第一轮不要求逐式推导或完整复现。':'入力 → 表現と融合 → 出力。最初から全数式の導出や実装の再現を目指す必要はない。',
'最后看证据':'最後に根拠を確認する',
'找一个支持结论的实验，再写一个仍没解决的问题。结果与解释分开。':'結論を支える実験と、残された課題を 1 つずつ挙げる。結果と解釈を分けて考える。',
'打开方式：':'開き方：',
'将整个压缩包解压，双击 <strong>index.html</strong>。论文通过相对路径打开，请保留旁边的 <strong>papers</strong> 文件夹。标注“来源（联网）”的链接仅供追溯，阅读正文无需联网。':'ZIP 全体を展開し、<strong>index.html</strong> をダブルクリックする。PDF は相対パスで開くため、同じ場所の <strong>papers</strong> フォルダも保持する。「出典（オンライン）」は参照用リンクで、本文の閲覧に通信は不要。',
'浏览器禁用了 JavaScript：论文、导读和折叠选读仍可阅读；搜索与进度记录不可用。':'JavaScript が無効です。論文・解説・発展文献は読めますが、検索と進捗記録は利用できません。',
'没有找到匹配项。可以试试“相机”“深度”“LSTM”，或清除搜索。':'一致する論文がありません。「カメラ」「深度」「LSTM」を試すか、検索をクリアしてください。',
'这里列出本包全部 {{COUNT}} 篇论文。主读用于建立路线，选读用于补充方法。收录版本可能是作者预印本，年份不等于 PDF 的上传日期。':'同梱した全 {{COUNT}} 本の一覧。基本文献で学習の流れをつかみ、発展文献で手法を補う。著者のプレプリントを含むため、掲載年と PDF のアップロード日は一致しない場合がある。',
'展开全部论文 · 点击名称直接打开本地文件':'全論文を表示 · 名前をクリックしてローカル PDF を開く',
'<th>阶段</th><th>顺序</th><th>论文</th><th>年份／版本</th><th>页数</th>':'<th>段階</th><th>区分</th><th>論文</th><th>年／版</th><th>ページ</th>',
'也可打开':'こちらからも閲覧：',
'论文阅读清单.txt':'reading_list_ja.txt',
'纯文本清单':'テキスト版リスト',
' 或 <a href="catalog.json"':' ／ <a href="catalog.json"',
'完整书目数据':'書誌データ',
'留下一页阅读记录':'読書メモを残す',
'不用复述整篇论文。写清楚任务、数据流和一个问题，就足够作为第一次组内交流的起点。':'論文全体を要約する必要はない。課題、データの流れ、疑問を 1 つ書けば、グループでの最初の議論につながる。',
'当前浏览器不允许保存本地文件页面的数据。勾选和笔记在本次打开期间仍有效；离开前请导出阅读记录。':'このブラウザではローカルページのデータを保存できません。開いている間はチェックとメモを利用できます。閉じる前に読書記録を書き出してください。',
'个人笔记（只保存在当前浏览器，不上传）':'個人メモ（このブラウザ内に保存し、外部には送信しません）',
'论文／阅读日期：&#10;1. 这篇要预测谁、解决什么问题？&#10;2. 输入是什么，哪些时间点可用？&#10;3. 输出在哪个坐标系，用什么单位？&#10;4. 画出：输入 → 中间表示 → 输出。&#10;5. 哪个实验最支持作者的观点？&#10;6. 一个局限或还没看懂的问题。':'論文／読んだ日：&#10;1. 誰を予測し、何を解決する研究か。&#10;2. 入力は何か。どの時刻の情報を使えるか。&#10;3. 出力の座標系と単位は何か。&#10;4. 入力 → 中間表現 → 出力を描く。&#10;5. 主張を最もよく支える実験はどれか。&#10;6. 限界や疑問を 1 つ挙げる。',
'导出阅读记录':'読書記録を書き出す',
'导入阅读记录':'読書記録を読み込む',
'阅读记录模板.txt':'reading_notes_template_ja.txt',
'下载文字模板':'メモ用テンプレート',
'移动目录或清理浏览器前，请先导出。':'フォルダの移動やブラウザのデータ削除前に書き出してください。',
'2026.10.21 · 组内学习版':'2026.10.21 · グループ学習用・日本語版',
'导读是原创学习提示，基于本地 PDF 的题名、摘要与相关方法说明整理，不代表所有方法均已复现，也不用于跨论文性能排名。PDF 保留原始署名及内容；版权与许可归原作者／出版方。公开再次分发前请核查各文献许可。':'本ガイドは同梱 PDF の題名・要旨・手法の記述を基にした独自の学習用解説。すべての手法の再現や、論文間の性能順位を示すものではない。PDF は原文・著者名を保持している。著作権と利用条件は原著者・出版者に帰属する。公開で再配布する場合は各文献の利用条件を確認すること。',
'没有 CDN、在线字体、远程图片或分析脚本。外部来源只有主动点击时才联网。':'CDN、オンラインフォント、外部画像、アクセス解析は使用していない。外部リンクをクリックしたときだけ通信する。',
'README_先读我.txt':'README_JA.txt',
'使用说明':'使い方',
'文件校验清单':'ファイル検証用ハッシュ',
'这不是本导读导出的有效记录。':'このガイドの有効な読書記録ではありません。',
'已保存在当前浏览器；可随时导出备份。':'このブラウザに保存しました。書き出してバックアップできます。',
"'找到 '+count+' 篇匹配论文（包含选读）'":"count+' 本の論文が見つかりました（発展文献を含む）'",
'已生成导出文件；请确认浏览器已保存。':'書き出しファイルを生成しました。ブラウザに保存されたことを確認してください。',
'记录文件过大。':'記録ファイルのサイズが大きすぎます。',
'请选择本导读导出的 reading-progress.json。':'このガイドから書き出した reading-progress.json を選択してください。',
'导入会替换当前阅读记录。继续吗？':'現在の読書記録を置き換えます。続けますか？',
'导入完成。':'読み込みが完了しました。',
'导入失败：':'読み込み失敗：',
}
template=Path(__file__).with_name('template.html').read_text(encoding='utf-8')
for a,b in T.items():
 assert a in template,a
 template=template.replace(a,b)
# Separate local storage by edition, while retaining the portable export format.
template=template.replace("const key='trajectory-reading-guide-v1';","const key='trajectory-reading-guide-v1';\nconst storageKey='trajectory-reading-guide-ja-v1';")
template=template.replace('localStorage.getItem(key)','localStorage.getItem(storageKey)').replace('localStorage.setItem(key,','localStorage.setItem(storageKey,')
# Japanese headings need slightly more room than their Chinese equivalents.
template=template.replace('grid-template-columns:60px 1fr','grid-template-columns:72px 1fr')
template=template.replace('font-size:48px;line-height:1.18','font-size:43px;line-height:1.35').replace('letter-spacing:-1.8px','letter-spacing:-.8px')
for key,val in {'NAV':nav,'SECTIONS':''.join(sections),'ROWS':rows,'COUNT':str(len(catalog)),'MAINCOUNT':'14','SIZE':f'{sum(c["bytes"] for c in catalog)/1048576:.0f}','DATA':json.dumps([{'id':c['id'],'title':c['title'],'stage':c['stage'],'level':c['level']} for c in catalog],ensure_ascii=False).replace('</',r'<\/')}.items():
 template=template.replace('{{'+key+'}}',val)
(PACK/'index.html').write_text(template,encoding='utf-8')
(PACK/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
readme='''画像認識から一人称視点の軌跡予測へ · オフライン論文ガイド
日本語版：2026-10-21

使い方
1. ZIP ファイル全体を任意のフォルダに展開してください。
2. index.html をダブルクリックし、Edge、Chrome、Firefox などで開きます。
3. 「PDF を開く」から各論文を読めます。
index.html だけを送らず、papers フォルダを含む ZIP 全体を共有してください。
Python のインストール、サーバーの起動、アカウント登録、インターネット接続は不要です。
PDF がプレビューされずダウンロードされる場合は、PC の PDF 閲覧ソフトで開いてください。

内容
27 本の論文 PDF、7 段階の日本語ガイド、書誌リスト、読書メモ用テンプレートを収録。
基本文献 14 本を表示し、残りは発展文献として折りたたんでいます。
検索は発展文献も対象です。解説と操作画面は日本語、論文 PDF は原文のままです。

読書記録
読了チェックとメモは、可能な場合に現在のブラウザ内へ保存します。
ローカルファイルの権限、フォルダの移動、ブラウザのデータ削除などで失われる場合があります。
保存・移行する場合は「読書記録を書き出す」を使ってください。
書き出した JSON は「読書記録を読み込む」で復元できます。中国語版の記録も読み込めます。
日本語版の自動保存領域は中国語版と分けています。情報の外部送信はありません。

出典と版
PDF は既存のローカル文献と、著者・学会・出版社が公開した版から収録しました。
内容・著者名は変更していません。一部はプレプリントです。
著作権と利用条件は原著者・出版者に帰属します。署名と出典を保持してください。
公開で再配布する場合は、各文献の利用条件を確認してください。
ガイドの説明や問いは独自の学習用解説であり、原著者の引用でも、全手法の再現報告でもありません。

オフライン動作
CDN、外部フォント、アクセス解析、外部画像を使用していません。
出典や補足教材の外部リンクをクリックした場合のみ通信します。
個人の講義資料、撮影動画、学習データ、アカウントのパスは含みません。

ファイル
index.html                     ガイドのトップページ
papers/                        論文 PDF（英数字のファイル名）
catalog.json                   書誌情報、日本語解説、PDF のハッシュ
reading_list_ja.txt             テキスト版の論文リスト
reading_notes_template_ja.txt   読書メモ用テンプレート
checksums.sha256                ファイルの SHA-256 ハッシュ
'''
(PACK/'README_JA.txt').write_text(readme,encoding='utf-8-sig')
lines=['論文リスト｜日本語版 2026-10-21','すべての論文は papers フォルダに収録。オフラインで閲覧できます。','']
for n,title,*_ in ST:
 lines += [f'{n:02d} {title}']
 for c in catalog:
  if c['stage']==n:lines += [f'  [{"基本" if c["level"]=="main" else "発展"}] {c["short"]} | {c["year"]}',f'  {c["title"]}',f'  ファイル：{c["file"]}',f'  出典：{c["ref"]}',f'  読む目的：{c["why"]}','']
(PACK/'reading_list_ja.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
(PACK/'reading_notes_template_ja.txt').write_text('論文：\n読者／日付：\n\n1. 何を解決する研究か。誰を予測するのか。\n2. 入力は何か。どの時刻の情報を利用できるか。\n3. 出力は何か。座標系と単位は何か。\n4. データの流れを 1 文または 1 枚の図で示す：\n5. 主張を最もよく支える実験：\n6. 限界や、まだ理解できていない点：\n7. 自分たちの研究とのつながり：\n',encoding='utf-8-sig')
print('Japanese guide built:',PACK)
