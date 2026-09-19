# レポート弱ラベル抽出

`modal_weak_labels.py` は Modal の A100 80GB 1枚で `google/gemma-4-31B-it` を動かし、レポートから固定長の構造化所見を抽出する。モデルは確率、信頼度、マスクを生成しない。JSONL が正本で、Parquet は分析用派生物である。

## 入出力

入力は UTF-8 の CSV とし、既定列は `StudyInstanceUID` と `report` である。患者データをリポジトリへ置かず、`rsna-weak-label-data` という非公開 Modal Volume に保存する。

```bash
uv run --with modal modal volume create rsna-weak-label-data
uv run --with modal modal volume put rsna-weak-label-data reports.csv /
```

モデル取得と推論は必ず別々に実行する。最初に `prepare-model` を実行する。この工程は GPU を一切割り当てない CPU 関数であり、Hugging Face の完全な snapshot と準備完了マニフェストを `rsna-weak-label-model-cache` Volume に保存する。

```bash
uv run --with modal modal run modal_weak_labels.py \
  --action prepare-model \
  --model google/gemma-4-31B-it
```

このコマンドは A100 関数を起動しない。再現性のため特定 revision を固定する場合、準備と推論の両方に同じ `--revision` を指定する。

2026年9月19日時点で、このモデルの Hugging Face リポジトリは公開されており、`gated` ではないため `HF_TOKEN` は不要である。将来アクセス制限が付いた場合は、Hugging Face 上で利用条件へ同意したアカウントの読み取りトークンを用意し、Modal Secret に `HF_TOKEN` として保存してから、準備関数へその Secret を割り当てる。トークンをコマンド行、ソース、Volumeへ直接保存しない。

準備完了後にのみ推論を起動する。次の操作は A100 80GB の課金を伴う。

```bash
uv run --with modal modal run modal_weak_labels.py \
  --action infer \
  --input-path reports.csv \
  --output-path outputs/raw_predictions.jsonl \
  --resume
```

`infer` はまず GPU なしの CPU 関数で準備完了マニフェストと snapshot の存在を確認する。未準備なら A100 80GB を要求する前に失敗する。GPU関数でも同じ確認を繰り返し、`HF_HUB_OFFLINE=1` と `TRANSFORMERS_OFFLINE=1` を設定し、マニフェスト内のローカル snapshot パスを vLLM に渡す。このため GPU 関数によるモデル取得は行われない。キャッシュ済み重みを A100 80GB へ転送する処理は vLLM の推論準備段階でのみ行う。

既定値はモデル `google/gemma-4-31B-it`、最大文脈長 8,192、最大生成長 1,200、`batch-size=32`、`max-num-seqs=32`、GPU メモリ利用率 0.92、接頭辞キャッシュ有効である。31Bの公式重みはBF16のまま読み込み、KVキャッシュ精度は `auto` としてvLLMのモデル既定値に従う。FP8 KVキャッシュと動的KVスケール計算は使用しない。さらにテキスト専用処理として画像と音声の入力枠を0にし、マルチモーダル事前計測のメモリを使わない。これは80GB 1枚に収めつつ、抽出精度を守るため重みの4ビット量子化を避ける構成である。メモリ不足時は、最初に `--max-model-len`、`--batch-size`、`--max-num-seqs` を下げる。

チャットは snapshot に同梱された Gemma 4 正式テンプレートを使う。`chat_template_kwargs={"enable_thinking": false}` を明示し、テンプレートが空の思考区画を生成開始位置へ付加した後、回答本文だけを構造化出力で生成する。独自テンプレートや reasoning parser は使わない。生成パラメータは snapshot の `generation_config.json` を vLLM の `generation_config="auto"` で読み込む。現在の公式設定は温度1.0、top-p 0.95、top-k 64、サンプリング有効であり、実際に解決された値を各JSONL行と指標JSONに保存する。

推論環境はvLLM 0.29.0とTransformers 5.17.0を明示的に固定する。構成根拠は [Gemma 4 モデルページ](https://huggingface.co/google/gemma-4-31B-it)、[vLLM 0.29.0 の対応モデル一覧](https://docs.vllm.ai/en/v0.29.0/models/supported_models/)、[vLLM Gemma 4 利用手順](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)、[Modal GPU 指定](https://modal.com/docs/guide/gpu) を参照する。

出力は同じ Volume の次の場所に保存される。

- `outputs/raw_predictions.jsonl`: 検証済み構造化出力と実際のモデル文字列、プロンプト版、モデル、推論設定
- `outputs/raw_predictions.errors.jsonl`: 推論失敗または不正出力。対象 UID、モデル文字列、例外を含む
- `outputs/raw_predictions.metrics.json`: 件数、トークン数、処理時間、処理量、GPU 最大メモリ、失敗数、意味警告数
- `outputs/raw_predictions.parquet`: 固定12対象を展開した分析用表

vLLM 自身の統計ログには接頭辞キャッシュ情報が出る。機械可読なヒット数を当該版のオフライン API から取得できない場合、指標 JSON の `prefix_cache_hit_metrics` は `null` のままとし、有効化状態を別項目で記録する。

## 安全な事前確認と再開

GPU を使わず入力列、重複 UID、固定プロンプト、再開対象を確認できる。

```bash
uv run python -m rsna2026.weak_label_infer \
  --input reports.csv \
  --output outputs/raw_predictions.jsonl \
  --dry-run
```

`--resume` は既存 JSONL の各行を再検証し、正常な UID のみをスキップする。不正な途中ファイルを黙って無視しない。失敗した UID は正本に入らないため、修正後の再実行で再試行される。

後処理だけを再実行する場合は次を使う。

```bash
uv run --with pyarrow python -m rsna2026.weak_label_postprocess \
  --input outputs/raw_predictions.jsonl \
  --output outputs/extracted_labels.parquet
```

確率変換は `weak_label_calibration.py` に隔離されている。58件の正解ラベルによる較正時はこの層だけを交換し、LLM 推論結果を再利用する。
