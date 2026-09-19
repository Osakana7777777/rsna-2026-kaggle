# レポート弱ラベル抽出

`modal_weak_labels.py` は Modal の H100 1枚で Qwen3.8 FP8 を動かし、レポートから固定長の構造化所見を抽出する。モデルは確率、信頼度、マスクを生成しない。JSONL が正本で、Parquet は分析用派生物である。

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
  --model Qwen/Qwen3.8-27B-FP8
```

このコマンドは H100 関数を起動しない。再現性のため特定 revision を固定する場合、準備と推論の両方に同じ `--revision` を指定する。

準備完了後にのみ推論を起動する。次の操作は H100 の課金を伴う。

```bash
uv run --with modal modal run modal_weak_labels.py \
  --action infer \
  --input-path reports.csv \
  --output-path outputs/raw_predictions.jsonl \
  --resume
```

`infer` はまず GPU なしの CPU 関数で準備完了マニフェストと snapshot の存在を確認する。未準備なら H100 を要求する前に失敗する。H100 関数でも同じ確認を繰り返し、`HF_HUB_OFFLINE=1` と `TRANSFORMERS_OFFLINE=1` を設定し、マニフェスト内のローカル snapshot パスを vLLM に渡す。このため GPU 関数によるモデル取得は行われない。キャッシュ済み重みを H100 へ転送する処理は vLLM の推論準備段階で行う。

既定値はモデル `Qwen/Qwen3.8-27B-FP8`、最大文脈長 65,536、最大生成長 1,200、GPU メモリ利用率 0.92、FP8 KV キャッシュ、接頭辞キャッシュ有効、思考無効である。生成パラメータはモデル snapshot の `generation_config.json` を vLLM の `generation_config="auto"` で読み込む。現在の公式 snapshot では温度 1.0、top-p 0.95、top-k 20 であり、実際に解決された値を各 JSONL 行と metrics JSON に保存する。`batch-size` と `max-num-seqs` はともに64を既定とし、Qwen3.8のMambaキャッシュ容量を超えるvLLM既定値1,024は使用しない。処理量の計測後に増やす場合も、起動ログに表示される利用可能なMambaキャッシュブロック数以下にする。

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
