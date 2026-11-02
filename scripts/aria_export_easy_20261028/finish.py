from pathlib import Path
import hashlib,json,zipfile,shutil,ast,sys
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path('E:/New_research/outputs/aria_export_easy_20261028')
PACK=ROOT/'Aria_Video_Exporter';AUDIT=ROOT/'audit'
TEST=AUDIT/'日本語 path & test'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipts=[(p,json.loads(p.read_text(encoding='utf-8'))) for p in (TEST/'出力 result').glob('*/result.json')]
complete=[(p,r) for p,r in receipts if r['status']=='complete'];failed=[(p,r) for p,r in receipts if r['status']=='failed']
assert len(complete)==2 and len(failed)==1
assert {r['downsample'] for p,r in complete}=={1,4}
for p,r in complete:
 assert (p.parent/'video.mp4').is_file() and r['verification']['full_decode_passed']
 assert r['verification']['metadata_matches_csv'] and r['verification']['frames']==75
for p,r in failed:assert not (p.parent/'video.mp4').exists()
for relative,h in json.loads((TEST/'preserved_first_run.json').read_text(encoding='utf-8')).items():
 assert sha(TEST/relative)==h,'Old export changed'
source=json.loads((AUDIT/'source_before.json').read_text(encoding='utf-8'))
assert sha(Path(source['file']))==source['sha256'],'Original VRS changed'
dependencies=json.loads((PACK/'DEPENDENCIES.json').read_text(encoding='utf-8'))
assert len(dependencies)==122
for d in dependencies:assert sha(PACK/d['file'])==d['sha256']
assert json.loads((PACK/'settings.json').read_text())['downsample']==1
for file in ('launcher.ps1','export_video.py','check_env.py','setup.sh','run.sh','system_setup.sh','requirements.lock'):
 assert sha(PACK/file)==sha(TEST/'Aria Video Exporter'/file),file+' differs from tested file'
for p in PACK.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
validation='''検証記録 / 验证记录 — 2026-10-28

PASS: Windows PowerShell から Ubuntu-24.04 の導入スクリプトを起動。
PASS: 新しい専用 Python 3.12 仮想環境へ、同梱の 122 wheel を --no-index / --require-hashes で導入。
PASS: pip check、公式 API と変換モジュールの import、FFmpeg と ffprobe。
PASS: パッケージ・VRS・出力のパスに日本語、空白、& を含む Windows → WSL の起動。
PASS: 実際の Aria Gen 2 VRS を公式 vrs_to_mp4 2.3.0 で変換。
  元データの RGB レコードは 76。出力動画は 75 フレーム、10 FPS、2560×1920、音声 1 ストリーム。
  コンテナの長さは約 7.655 秒。元レコードと動画フレーム数の相違は公式の時刻サンプリング条件による。
PASS: 75 フレームを全編デコードし、CSV 75 行と MP4 内埋め込みの VRS 時刻列が完全一致。
PASS: 縮小率 4 で 640×480 を出力し、同じデコード・時刻検査に合格。
PASS: バッチの先頭に不正な VRS を入れても、後続の正常なファイルを処理。
  失敗分には failed の結果とログを保存し、完成扱いの video.mp4 は作らない。
PASS: 再実行前後で既存の全出力ファイルと元 VRS の SHA-256 が不変。
PASS: 04_CHECK と同じ Check モードの Windows 起動。
PASS: PowerShell 構文、Bash 構文、Python 構文、ローカル説明ページのリンク。
PASS: 配布 ZIP の CRC、全ファイルの SHA-256、別フォルダへの展開後の一致。

範囲外 / 限界
・WSL 未導入の別 PC での初回有効化、BIOS、管理者権限、再起動は未検証。
・実行ホストの既存 WSL2 / Ubuntu 24.04 上で、新規 Python 環境を作成して検証。
・ファイル選択ダイアログの手操作は自動化していない。同じランチャーの明示パス引数で実変換を検査。
・全種類の VRS、音声設定、Windows ARM、Mac、ネットワーク共有パスの互換性を保証しない。
・導出動画は研究用の無損失フレームダンプではない。元 VRS と時刻 CSV を保持すること。
・パックにテスト用の録画、出力動画、ユーザー環境やアカウント情報は同梱していない。

中文摘要：已用全新独立 Python 环境实际跑通安装、官方导出、日文/空格路径、
完整解码、时间戳核对、批量坏文件继续执行和旧结果保护。组员电脑首次启用 WSL
所需的权限、虚拟化与重启，仍需在该电脑上完成。
'''
(PACK/'VALIDATION.txt').write_text(validation,encoding='utf-8-sig')
class Links(HTMLParser):
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'href' in a:
   u=urlsplit(a['href'])
   if not u.scheme:assert (PACK/unquote(u.path)).is_file()
  assert not(tag=='script' or 'src' in a),'Unexpected remote resource'
Links().feed((PACK/'START_HERE.html').read_text(encoding='utf-8'))
files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='checksums.sha256')
assert not any(p.suffix.lower() in ('.vrs','.mp4','.log') or 'run_logs' in p.parts for p in files)
for p in files:
 if p.suffix in ('.py','.ps1','.sh','.txt','.html','.json','.cmd'):
  txt=p.read_text(encoding='utf-8-sig')
  assert 'nytha' not in txt and 'E:/New_research' not in txt and 'E:\\New_research' not in txt,p
(PACK/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),encoding='utf-8')
archive=ROOT/'Aria_Video_Exporter_Windows_JA_ZH.zip'
with zipfile.ZipFile(archive,'w') as z:
 for p in sorted(PACK.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.whl' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert all(n.startswith(PACK.name+'/') and '..' not in Path(n).parts for n in z.namelist())
 z.extractall(AUDIT/'zip_roundtrip')
for p in PACK.rglob('*'):
 if p.is_file():assert sha(p)==sha(AUDIT/'zip_roundtrip'/p.relative_to(ROOT))
result={'package':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
 'wheel_count':len(dependencies),'complete_exports':[r['verification'] for p,r in complete],
 'expected_invalid_file_failures':len(failed),'source_and_old_outputs_unchanged':True,
 'windows_wsl_path_test':True,'fresh_python_env_offline_install':True,'pip_check':True,
 'archive_crc_and_relocated_hashes':True,'no_recordings_or_private_runtime_in_package':True,
 'not_tested':['Fresh WSL Windows feature activation and restart on recipient PC','Manual GUI dialog interaction','Windows ARM/macOS']}
(AUDIT/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
report=Path('E:/New_research/reports/aria_export_easy_20261028');report.mkdir(exist_ok=True)
(report/'RESULTS.txt').write_text(validation+'\nArchive SHA-256: '+result['sha256']+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
