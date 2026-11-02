日本語版・研究引継ぎ文書
作成日：2026-11-03。研究結果の基準日：2026-10-29。

人が読む場合
まず01（現在の全体像）、02（実験の進捗）、07（今後の候補）を読む。
必要に応じて03（データ）、04（短期）、05（長期）、06（学習診断）を参照する。

AIに渡す場合
最初にAI_HANDOFF_JA.txtを渡し、同じフォルダの01〜07も参照できるようにする。
AI_HANDOFF_JA.txtは実験の自動実行指示ではなく、事実・制約・入口の引継ぎである。

選定7文書と中国語原文の対応（プロジェクトルートからの相対パス）
01_PROJECT_OVERVIEW_JA.txt
  reports/github_share_inventory_20261103/PROJECT_OVERVIEW.txt
02_FORECAST_MILESTONES_JA.txt
  reports/forecast_milestones_20261029/RESULTS.txt
03_G1_DATA_CONSTRUCTION_JA.txt
  reports/g1_bulk_association_20261029/RESULTS.txt
04_G1_EXPANDED_FORECAST_JA.txt
  reports/g1_expanded_forecast_20261029/RESULTS.txt
05_LONG_HORIZON_JA.txt
  reports/horizon_15_15_g1_20261029/RESULTS.txt
06_LEARNING_DIAGNOSIS_JA.txt
  reports/tract_learning_map_20261029/RESULTS.txt
07_RESEARCH_BRANCHES_JA.txt
  reports/research_branches_20261029/OVERVIEW.txt

翻訳方針
選定7文書は要約版ではなく、原文の実験結果、数値、比較条件、制約、検証記録を日本語化したもの。
原報告時点を維持し、共有に含まれないファイルや旧環境パスには注記を加えた。
箇条書きや表、JSON形式の一部は読める形へ整形したが、実験条件や数値の意味は変更していない。
G1データ構築直後の「未学習」と、その後の拡充学習結果は異なる時点の記録であり矛盾しない。
注釈＝アノテーション、対応付け＝人物IDのassociation、収録＝recording、窓＝window。
GT＝ground truth。自動参照注釈や元source-IDを、人手の真の正解に読み替えない。

このフォルダに原データ・重み・outputsは同梱していない。文書が指すパスの存在は受領した共有物で確認する。
翻訳の確認情報と原文／日本語版のSHA256はTRANSLATION_MANIFEST.jsonにある。
