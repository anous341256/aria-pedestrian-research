from pathlib import Path
import email,hashlib,json,zipfile,html
ROOT=Path('E:/New_research/outputs/aria_export_easy_20261028')
PACK=ROOT/'Aria_Video_Exporter'
rows=[];lock=[]
for w in sorted((PACK/'wheels').glob('*.whl')):
 with zipfile.ZipFile(w) as z:
  meta=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
 h=hashlib.sha256(w.read_bytes()).hexdigest();name=str(meta['Name']);version=str(meta['Version'])
 lock.append(f'{name}=={version} --hash=sha256:{h}')
 rows.append(dict(name=name,version=version,file='wheels/'+w.name,sha256=h,bytes=w.stat().st_size,
  source='https://pypi.org/project/'+name+'/'+version+'/',
  license=str(meta.get('License-Expression') or meta.get('License') or 'See original license files inside wheel')))
(PACK/'requirements.lock').write_text('\n'.join(lock)+'\n',encoding='utf-8')
(PACK/'DEPENDENCIES.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
JA='''Aria VRS → MP4 かんたん変換パック
2026-10-28 / Windows x64 + WSL2 + Ubuntu 24.04

最初に START_HERE.html を開いてください。
このパックは研究グループが作成した起動・導入補助ツールです。
Meta 公式の未変更の vrs_to_mp4（Project Aria Tools 2.3.0）を呼び出します。
Meta 公式配布の Windows アプリではありません。

【必要なもの】
・Intel/AMD 64 ビットの Windows 11、または Windows 10 2004 / Build 19041 以降。
・初回はインターネット接続、WSL を有効化するための管理者権限と、場合によって再起動。
・仮想化が有効な PC。Windows ARM、Mac、Ubuntu 22.04 向けのパッケージではありません。
・目安として 10 GB 以上の空き容量に加えて、出力動画を保存する領域を確保してください。
・Windows 側の Python や CUDA、GPU、Meta ログインは不要です。

【初回だけ】
1. ZIP 全体をローカルドライブに展開します。ZIP 内から直接起動しないでください。
   例：C:\\AriaExport。フォルダ内のファイルと wheels フォルダをすべて保持してください。
2. WSL / Ubuntu-24.04 がない場合は、01_INSTALL_WSL.cmd を右クリックし、
   「管理者として実行」します。Windows の指示があれば再起動してください。
   すでに Ubuntu-24.04 が使える場合は、この手順を省略します。
3. スタートメニューから Ubuntu 24.04 を一度起動し、ユーザー名とパスワードの設定を完了します。
   パスワード入力中に文字が表示されないのは正常です。
4. Windows に戻り、02_SETUP.cmd をダブルクリックします。
   Ubuntu の python3.12-venv と FFmpeg を apt で取得し、同梱の Python 依存関係を
   専用の仮想環境へインストールします。通常の Windows 管理者実行は不要です。
   Linux のシステム依存関係の追加だけは WSL 内の root で行います。
   READY / 準備完了 と出れば導入完了です。初回のシステム導入には通信が必要です。

【毎回の変換】
5. 03_EXPORT.cmd をダブルクリックします。
6. .vrs ファイルを選びます。複数選択できます。次に保存先フォルダを選びます。
7. DONE / 完了 が出るまで待ちます。CPU で実行できます。処理中はウィンドウを閉じないでください。
   変換中は約 15 秒ごとに経過時間を表示します。終了後、出力先フォルダが開きます。
   原本 VRS は読み取るだけです。毎回別フォルダを作り、既存の出力を上書きしません。

【出力】
video.mp4                         RGB 映像。録音がある場合は音声も含みます。
logs/mp4_to_vrs_time_ns.csv        MP4 時刻と VRS デバイス時刻の対応
logs/vrs_to_mp4_log.json           公式ツールのスキップ・重複等の診断
logs/export.log                   公式ツールの詳細ログ
logs/decode_check.log             全編デコード検査ログ
result.json                       成否、解像度、フレーム数、時刻照合の検査結果
失敗時は video.partial.mp4 が残る場合があります。完了動画として使用しないでください。

【映像の仕様】
RGB ストリーム 214-1、縮小率 1（元の解像度）、公式ツールの既定の音声チャンネル 0/2。
魚眼の歪み補正や MPS、IMU、深度の抽出は行いません。
公式変換は公称 FPS の動画時刻に近いフレームを選ぶため、重複・スキップがあり得ます。
MP4 のフレーム番号を元 VRS のレコード番号と同一視せず、CSV の対応を利用してください。
必要な場合は settings.json の downsample を 2 または 4 にすると縦横の解像度が縮小します。
環境導入後の変換はオフラインで可能です。この起動補助ツールにアップロード処理はありません。

【困ったとき】
・ウィンドウが閉じる／失敗する：04_CHECK.cmd を実行し、run_logs 内のログを確認します。
・WSL がない／Ubuntu がない：手順 2〜3 を完了し、Windows を再起動してやり直します。
・WSL エラー 0x80370102：仮想化と Virtual Machine Platform、再起動を確認します。
・ダウンロードが 0% で止まる：管理者 PowerShell で次を試します。
  wsl --install --web-download -d Ubuntu-24.04
・apt が失敗する：VPN、プロキシ、ネットワーク、ディスク容量を確認し、02_SETUP を再実行します。
・管理対象 PC で WSL やスクリプトが禁止：管理者へ利用許可・導入を依頼してください。
・RGB がない VRS：このパックでは変換できません。収録時の RGB 設定と VRS を確認してください。
・パス：ローカルドライブを使用してください。共有フォルダ/UNC、ZIP 内からの実行は非対応です。
・powershell の起動は、そのプロセスだけ ExecutionPolicy Bypass を指定します。
  永続的な実行ポリシーや Windows Defender の設定は変更しません。
・Linux 環境の既定パス：~/.local/share/aria-video-easy/venv-2.3.0-20261028
  他の Python 環境は変更しません。環境の再実行は専用マーカーを持つ場所に限ります。
ログにはローカルのパスやファイル名が含まれます。相談時に共有する範囲は確認してください。

【検証の範囲】
同梱の依存関係から新しい Python 仮想環境を作り、実データの変換を確認しています。
Windows での起動、空白・日本語を含むパス、タイムスタンプ照合、動画全編のデコードを検査。
ただし、WSL 未導入の別 PC での管理者権限・BIOS・再起動は、その PC での確認が必要です。
具体的な検査結果は VALIDATION.txt を参照してください。

【出典】
https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/projectariatools/installation
https://github.com/facebookresearch/projectaria_tools
https://learn.microsoft.com/windows/wsl/install
DEPENDENCIES.json に同梱パッケージの版・出典・ハッシュを記録しています。
'''
ZH='''Aria VRS → MP4 视频导出懒人包
2026-10-28；Windows x64 + WSL2 + Ubuntu 24.04

这是我们制作的安装/启动封装，内部调用未修改的 Meta 官方 vrs_to_mp4（Project Aria Tools 2.3.0）。
它不是 Meta 发布的 Windows 原生应用。日语图文说明见 START_HERE.html。

1. 完整解压 ZIP 到本地磁盘，例如 C:\\AriaExport，保留 wheels 目录。
2. 若未安装 WSL 和 Ubuntu-24.04：右键 01_INSTALL_WSL.cmd → 以管理员身份运行。
   按系统提示重启。已经能用 Ubuntu-24.04 的人跳过这一步。
3. 从开始菜单打开 Ubuntu 24.04，完成首次用户名、密码设置。
4. 回到 Windows，双击 02_SETUP.cmd。等待 READY。
   Ubuntu 的 Python venv 与 FFmpeg 由 apt 联网安装；Python 依赖从包内离线安装。
5. 双击 03_EXPORT.cmd，选择一个或多个 VRS，再选择输出目录，等待 DONE。
6. 报错时运行 04_CHECK.cmd，查看 run_logs；转换错误还可查看输出目录的 logs/export.log。

要求：Intel/AMD 64 位 Windows 11 或 Windows 10 2004 / Build 19041 以上。
首次启用 WSL 需要管理员权限、虚拟化支持，可能需要重启；无法用压缩包绕过这些系统要求。
本包不适用于 Windows ARM、Mac 或 Ubuntu 22.04。Windows 侧不需要 Python、CUDA 或 Meta 登录。
建议预留至少 10 GB 加视频输出空间。首次安装系统组件需联网，之后导出可离线进行。

输出：video.mp4、logs/mp4_to_vrs_time_ns.csv、logs/vrs_to_mp4_log.json、转换日志、result.json。
默认 RGB 214-1，不缩小、不去畸变；录制有音频时使用官方默认音频通道 0/2。
官方工具按公称 FPS 的时间选取邻近帧，可能出现重复或跳帧；不要把 MP4 帧号直接当 VRS 索引。
每次创建新目录，不覆盖旧结果，不修改原 VRS，不上传原始数据。
完成前会逐帧解码检查，并核对视频帧数、CSV 映射和 MP4 内嵌时间戳。
失败结果会保留为 partial 和日志，不会伪装成成功视频。

settings.json 可把 downsample 改成 2 或 4，按比例缩小画面。
本包仅做普通视频导出，不自动处理 MPS、IMU、深度，也不是 Rerun 多传感器查看器。
Python 环境放在 WSL 用户目录 ~/.local/share/aria-video-easy/venv-2.3.0-20261028。
系统依赖安装在指定 Ubuntu 内，其他 Python 环境不变。
PowerShell 的 Bypass 仅针对本次启动进程，不永久修改执行策略。

常见问题：
WSL 0x80370102：检查 BIOS 虚拟化、Virtual Machine Platform 和重启。
WSL 下载卡 0%：管理员 PowerShell 执行 wsl --install --web-download -d Ubuntu-24.04。
apt 下载失败：检查 VPN/代理/网络/磁盘，再运行 02_SETUP。
公司/学校系统策略禁止 WSL 或脚本：需要本机管理员处理。
路径需本地磁盘；不支持网络 UNC 路径或直接在 ZIP 中启动。
日志包含文件名和本地路径，发给别人排错前注意检查分享范围。

已测试现有 WSL 上的全新独立 Python 环境和真实 VRS 导出，具体见 VALIDATION.txt。
没有替组员验证其电脑首次启用 WSL、BIOS、管理员权限或重启，不能保证这部分无障碍。
'''
(PACK/'README_JA.txt').write_text(JA,encoding='utf-8-sig')
(PACK/'README_ZH.txt').write_text(ZH,encoding='utf-8-sig')
page='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aria 動画変換パック</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f2;color:#20393e;font:16px/1.85 "Yu Gothic UI",Meiryo,sans-serif}.wrap{max-width:940px;margin:auto;padding:45px 26px}header{padding:30px 0}h1{font-size:38px;line-height:1.4;margin:12px 0}h2{font-size:23px;margin-top:35px}.tag{color:#197268;font-weight:bold;font-size:12px;letter-spacing:2px}.lead{color:#50666a}.steps{display:grid;gap:16px}.step{background:white;border:1px solid #d7e3de;border-radius:12px;padding:22px;display:grid;grid-template-columns:44px 1fr;gap:15px}.num{font-size:30px;color:#197268}h3{margin:0 0 7px;font-size:19px}.step p{margin:0}code{background:#eaf1ed;padding:3px 7px;border-radius:4px;overflow-wrap:anywhere}a{color:#147168}.note{background:#fff0d9;padding:20px;border-radius:9px;margin:25px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}details{background:white;border:1px solid #d7e3de;padding:17px;border-radius:9px;margin:12px 0}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.9 inherit}footer{border-top:1px solid #d7e3de;padding-top:22px;font-size:12px;color:#637676}@media(max-width:600px){h1{font-size:29px}.grid{grid-template-columns:1fr}.wrap{padding:22px 17px}}</style></head><body><main class="wrap">
<header><div class="tag">ARIA GEN 2 · WINDOWS x64 · 日本語 / 中文</div><h1>VRS を選んで、<br>MP4 に変換する。</h1><p class="lead">Meta 公式の変換ツールを、Windows から簡単に起動するための研究グループ向けパック。<br>導入後はファイルを選ぶだけ。録画データは PC 内で処理します。</p></header>
<div class="note"><b>初回だけ WSL が必要です。</b><br>対応：Intel/AMD x64 の Windows 10 2004 以降 / Windows 11 ＋ Ubuntu 24.04。初回のシステム導入には通信、管理者権限、場合によって再起動が必要です。Windows ARM / Mac 向けではありません。GPU・CUDA・Windows 版 Python は不要です。</div>
<h2>最短の使い方</h2><div class="steps">
<div class="step"><span class="num">01</span><div><h3>ZIP 全体を展開</h3><p>例：<code>C:\\AriaExport</code>。ZIP 内から直接起動せず、<code>wheels</code> を含む全ファイルを同じフォルダに保管します。</p></div></div>
<div class="step"><span class="num">02</span><div><h3>WSL と Ubuntu を導入（初回のみ）</h3><p><code>01_INSTALL_WSL.cmd</code> を右クリック → <b>管理者として実行</b>。必要なら再起動し、スタートメニューから Ubuntu 24.04 を開いてユーザー設定を完了します。すでに使える場合は省略できます。</p></div></div>
<div class="step"><span class="num">03</span><div><h3>変換環境を作る（初回のみ）</h3><p><code>02_SETUP.cmd</code> をダブルクリック。<b>READY / 準備完了</b> が出るまで待ちます。Ubuntu のシステム依存関係はネットから、Python 依存関係は同梱ファイルから導入します。</p></div></div>
<div class="step"><span class="num">04</span><div><h3>VRS と出力先を選ぶ</h3><p><code>03_EXPORT.cmd</code> をダブルクリック → VRS を選択 → 保存先を選択。複数ファイルも選べます。<b>DONE / 完了</b> になれば、出力フォルダが開きます。</p></div></div></div>
<h2>得られるもの</h2><div class="grid"><div><h3>動画と時刻の対応</h3><p><code>video.mp4</code> は RGB 映像です。録音がある場合は音声も含みます。<code>logs/mp4_to_vrs_time_ns.csv</code> に元 VRS との時刻対応を保存します。</p></div><div><h3>確認結果とログ</h3><p>全編のデコードと時刻対応をチェックし、<code>result.json</code> に結果を記録します。失敗時は途中のファイルとログを残します。</p></div></div>
<div class="note">元 VRS は変更せず、出力は毎回新しいフォルダに作成します。映像は既定で元の解像度・魚眼のままです。公式ツールは公称 FPS に合わせてフレームを選ぶため、MP4 のフレーム番号と VRS のレコード番号は必ずしも一致しません。研究用途では CSV を使って対応付けてください。</div>
<h2>困ったとき</h2><p><code>04_CHECK.cmd</code> で環境を診断できます。<code>run_logs</code> と、各出力の <code>logs/export.log</code> を確認してください。</p>
<details><summary>日本語の詳しい手順・トラブル対処</summary><pre>JATEXT</pre></details>
<details lang="zh-CN"><summary>中文使用说明</summary><pre>ZHTEXT</pre></details>
<p><a href="README_JA.txt">日本語テキスト</a> ／ <a href="README_ZH.txt">中文文本</a> ／ <a href="VALIDATION.txt">検証記録</a> ／ <a href="DEPENDENCIES.json">依存関係一覧</a></p>
<footer>2026-10-28 · 起動補助は研究グループ作成。Meta 公式の Windows アプリではありません。<br><a href="https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/projectariatools/installation">Meta 公式の対応環境</a> · <a href="https://learn.microsoft.com/windows/wsl/install">Microsoft WSL インストール手順</a><br>この案内ページはオフラインで読めます。外部リンクはクリックしたときだけ通信します。</footer>
</main></body></html>'''
(PACK/'START_HERE.html').write_text(page.replace('JATEXT',html.escape(JA)).replace('ZHTEXT',html.escape(ZH)),encoding='utf-8')
(PACK/'THIRD_PARTY_NOTICES.txt').write_text('''This package contains original, unmodified Python wheels obtained from PyPI.
Project Aria Tools is provided by Meta under Apache-2.0.
The Windows launchers, shell scripts and export wrapper are group-authored conveniences,
not an official Meta Windows product. The official converter is not modified.
Each wheel retains its original metadata and license files; consult the .dist-info
directory inside that wheel. DEPENDENCIES.json records names, versions, sources and hashes.
FFmpeg and Ubuntu system components are installed from the recipient Ubuntu repositories;
they are not redistributed here as system binaries. imageio-ffmpeg is an unmodified
upstream wheel; its bundled binaries and notices remain within that wheel.
We do not bundle any WSL disk, user account, credentials, recording, or experimental data.
The convenience wrapper does not invoke Jupyter servers, Rerun servers or cloud MPS.
''',encoding='utf-8')
print('Documentation and lock generated:',len(rows),'wheels')
