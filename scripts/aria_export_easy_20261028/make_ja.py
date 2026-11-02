from pathlib import Path
import shutil,json,hashlib,re,ast
ROOT=Path('E:/New_research/outputs/aria_export_easy_20261028')
SRC=ROOT/'Aria_Video_Exporter';DST=ROOT/'Aria_Video_Exporter_JA';AUDIT=ROOT/'audit_ja';AUDIT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(AUDIT/'original_hashes.json').write_text(json.dumps({p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file()},indent=2),encoding='utf-8')
DST.mkdir(exist_ok=True)
for p in SRC.rglob('*'):
 if p.is_file() and p.name not in ('README_ZH.txt','checksums.sha256'):
  q=DST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def replace_file(name,replacements,bom=False):
 p=DST/name;s=p.read_text(encoding='utf-8-sig')
 for a,b in replacements.items():
  assert a in s,(name,a)
  s=s.replace(a,b)
 p.write_text(s,encoding='utf-8-sig' if bom else 'utf-8',newline='\n' if p.suffix=='.sh' else None)
replace_file('launcher.ps1',{
 'Invalid distribution name in settings.json.':'settings.json のディストリビューション名が無効です。',
 'Working... / Processing: ':'処理中です。経過時間：',
 "+' s'":"+' 秒'",
 'WSL command failed (exit $code). Logs: $prefix.*.log':'WSL の処理に失敗しました（終了コード $code）。ログ：$prefix.*.log',
 'Use a local Windows drive, e.g. C:\\AriaExport. UNC/network paths are not supported.':'C:\\AriaExport などのローカルドライブを使用してください。ネットワーク共有パスは非対応です。',
 'Aria Video Exporter | Project Aria Tools 2.3.0':'Aria 動画変換ツール・日本語版 | Project Aria Tools 2.3.0',
 'This package requires an Intel/AMD x64 Windows PC (not Windows ARM).':'Intel/AMD x64 の Windows PC が必要です。Windows ARM は非対応です。',
 "   Write-Host 'Please right-click 01_INSTALL_WSL.cmd and choose Run as administrator.'\n":'',
 'Installing WSL + Ubuntu-24.04. Follow Windows prompts; restart if requested.':'WSL と Ubuntu-24.04 を導入します。Windows の指示に従い、必要に応じて再起動してください。',
 'WSL installation did not finish. See START_HERE.html troubleshooting.':'WSL の導入が完了していません。START_HERE.html のトラブル対処を確認してください。',
 'After restart, open Ubuntu-24.04 once and finish username/password setup. Then run 02_SETUP.cmd.':'再起動後、Ubuntu-24.04 を開いてユーザー名とパスワードを設定し、02_SETUP.cmd を実行してください。',
 'WSL missing. Run 01_INSTALL_WSL.cmd first.':'WSL が見つかりません。先に 01_INSTALL_WSL.cmd を実行してください。',
 'Step 1/2: Ubuntu system prerequisites (network required).':'手順 1/2：Ubuntu のシステム依存関係を導入します（通信が必要です）。',
 'Step 2/2: Installing bundled Python dependencies (offline).':'手順 2/2：同梱の Python 依存関係を導入します（オフライン）。',
 'READY / 準備完了. Next: 03_EXPORT.cmd':'準備完了。次に 03_EXPORT.cmd を実行してください。',
 'CHECK PASSED. Diagnostic logs: $logRoot':'環境チェックに合格しました。診断ログ：$logRoot',
 'Aria recordings (*.vrs)|*.vrs':'Aria 録画ファイル (*.vrs)|*.vrs',
 'VRS ファイルを選択 / Select VRS files':'VRS ファイルを選択してください（複数選択可）',
 '出力先フォルダを選択 / Select output folder':'出力先フォルダを選択してください',
 'Cancelled.':'キャンセルしました。',
 'Not a VRS file: $inputPath':'VRS ファイルではありません：$inputPath',
 'Exporting RGB video. Downsample factor: ':'RGB 動画を変換します。縮小率：',
 '. Audio if present.':'。録音がある場合は音声も含みます。',
 'DONE / 完了: $OutputRoot':'変換が完了しました：$OutputRoot',
 'ERROR: $message':'エラー：$message',
 'See START_HERE.html. Keep run_logs for troubleshooting.':'START_HERE.html を確認してください。原因調査のため run_logs を保存してください。',
},True)
replace_file('export_video.py',{
 'ffprobe failed: ':'ffprobe による検査に失敗しました：',
 'Expected one video stream.':'動画ストリームが 1 つであることを確認できません。',
 'Empty exported video.':'出力動画にフレームがありません。',
 'Timestamp validation failed: video={count}, CSV={len(rows)}, metadata={len(embedded)}':'時刻照合に失敗しました：動画={count}、CSV={len(rows)}、メタデータ={len(embedded)}',
 'Non-monotonic timestamp mapping.':'時刻対応が単調増加の条件を満たしていません。',
 'Full video/audio decoding failed; see decode_check.log.':'動画または音声の全編デコードに失敗しました。decode_check.log を確認してください。',
 'Select an existing .vrs file.':'存在する .vrs ファイルを選択してください。',
 'downsample must be 1, 2 or 4.':'downsample は 1、2、4 のいずれかにしてください。',
 "'EXPORT:'":"'変換対象：'", "'OUTPUT:'":"'出力先：'",
 'Please wait. Official progress is saved in logs/export.log.':'しばらくお待ちください。公式ツールの詳細は logs/export.log に保存します。',
 'Official converter exited {p.returncode}; see logs/export.log.':'公式変換ツールが終了コード {p.returncode} で停止しました。logs/export.log を確認してください。',
 'Checking video and timestamp mapping...':'動画とタイムスタンプの対応を検査しています…',
 'Input file changed during conversion.':'変換中に入力ファイルが変更されました。',
 "'SUCCESS:'":"'変換成功：'",
})
replace_file('check_env.py',{'Python 3.12 required':'Python 3.12 が必要です。','Linux x86_64 required':'Linux x86_64 が必要です。',"name+' not installed'":"name+' が見つかりません。02_SETUP.cmd を実行してください。'"})
replace_file('system_setup.sh',{'Ubuntu 24.04 required.':'Ubuntu 24.04 が必要です。','x86_64 required; ARM is not supported by this package.':'x86_64 が必要です。このパックは ARM に対応していません。','Run the Windows 02_SETUP.cmd launcher.':'Windows から 02_SETUP.cmd を実行してください。','Installing Ubuntu prerequisites (network required)...':'Ubuntu の依存関係を導入しています（通信が必要です）…'})
replace_file('setup.sh',{'Ubuntu 24.04 x86_64 required.':'Ubuntu 24.04 x86_64 が必要です。','Using the configured root WSL user; only the dedicated virtual environment is installed here.':'現在の WSL ユーザーは root です。この処理では専用の仮想環境を作成します。','Refusing to change an unrecognized environment: $env_dir':'このパックの管理対象ではない環境のため、変更を中止しました：$env_dir','READY. Next: double-click 03_EXPORT.cmd in Windows.':'準備完了。Windows で 03_EXPORT.cmd をダブルクリックしてください。'})
replace_file('run.sh',{'Run 02_SETUP.cmd first.':'先に 02_SETUP.cmd を実行してください。','Invalid action.':'操作の指定が無効です。'})
for p in DST.glob('*.cmd'):
 s=p.read_text(encoding='ascii').replace('@echo off\n','@echo off\nchcp 65001 >nul\n').replace('echo Exit code: %RESULT%','echo 終了コード：%RESULT%')
 p.write_bytes(s.replace('\n','\r\n').encode('utf-8'))
p=DST/'START_HERE.html';s=p.read_text(encoding='utf-8')
s=s.replace('日本語 / 中文','日本語版').replace('READY / 準備完了','準備完了').replace('DONE / 完了','変換が完了しました')
s,n=re.subn(r'<details lang="zh-CN">.*?</details>','',s,flags=re.S);assert n==1
s=s.replace(' ／ <a href="README_ZH.txt">中文文本</a>','')
s=s.replace('<h2>困ったとき</h2>','<p>起動補助の案内とエラー説明は日本語です。Meta 公式ツール・pip・Ubuntu の詳細ログと、機械処理用 JSON のキーは原文のままです。</p><h2>困ったとき</h2>')
p.write_text(s,encoding='utf-8')
replace_file('README_JA.txt',{'READY / 準備完了':'準備完了','DONE / 完了':'変換が完了しました','2026-10-28 / Windows x64':'日本語専用版 2026-10-28 / Windows x64'},True)
with (DST/'README_JA.txt').open('a',encoding='utf-8') as f:f.write('\n起動補助の案内・エラー説明は日本語です。公式ツール、pip、Ubuntu のログと JSON キーは原文を保持します。\n')
p=DST/'VALIDATION.txt';s=p.read_text(encoding='utf-8-sig').split('\n中文摘要：')[0].replace('検証記録 / 验证记录','検証記録・日本語版')
p.write_text('以下は共通処理の検証記録です。日本語版での再検証は末尾に記載します。\n\n'+s,encoding='utf-8-sig')
for p in DST.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Japanese edition built:',DST)
