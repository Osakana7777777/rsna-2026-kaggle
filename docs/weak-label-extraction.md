# レポート弱ラベル抽出

`modal_weak_labels.py` は Modal の A100 80GB 1枚で `google/gemma-4-12B-it` を動かし、レポートから固定長の構造化所見を抽出する。モデルは確率、信頼度、マスクを生成しない。JSONL が正本で、Parquet は分析用派生物である。

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
  --model google/gemma-4-12B-it
```

このコマンドは A100 関数を起動しない。再現性のため特定 revision を固定する場合、準備と推論の両方に同じ `--revision` を指定する。

2026年9月19日時点で、このモデルの Hugging Face リポジトリは公開されており、`gated` ではないため `HF_TOKEN` は不要である。将来アクセス制限が付いた場合は、Hugging Face 上で利用条件へ同意したアカウントの読み取りトークンを用意し、Modal Secret に `HF_TOKEN` として保存してから、準備関数へその Secret を割り当てる。トークンをコマンド行、ソース、Volumeへ直接保存しない。

準備完了後にのみ推論を起動する。次の操作は A100 80GB の課金を伴う。

```bash
uv run --with modal modal run modal_weak_labels.py \
  --action infer \
  --input-path reports.csv \
  --output-path outputs/raw_predictions.jsonl \
  --thinking \
  --resume
```

`infer` はまず GPU なしの CPU 関数で準備完了マニフェストと snapshot の存在を確認する。未準備なら A100 80GB を要求する前に失敗する。GPU関数でも同じ確認を繰り返し、`HF_HUB_OFFLINE=1` と `TRANSFORMERS_OFFLINE=1` を設定し、マニフェスト内のローカル snapshot パスを vLLM に渡す。このため GPU 関数によるモデル取得は行われない。キャッシュ済み重みを A100 80GB へ転送する処理は vLLM の推論準備段階でのみ行う。

既定値はモデル `google/gemma-4-12B-it`、最大文脈長 8,192、最大生成長 4,096、`batch-size=32`、`max-num-seqs=32`、GPU メモリ利用率 0.92、接頭辞キャッシュ有効である。12Bの公式重みはBF16のまま読み込み、KVキャッシュ精度は `auto` としてvLLMのモデル既定値に従う。FP8 KVキャッシュと動的KVスケール計算は使用しない。さらにテキスト専用処理として画像と音声の入力枠を0にし、マルチモーダル事前計測のメモリを使わない。

固定抽出指示と症例入力は、1個のユーザーメッセージへ連結する。固定抽出指示を必ず先頭、StudyInstanceUIDとレポートを末尾に置くため、全症例で同一の長いトークン接頭辞を共有し、vLLMの接頭辞キャッシュを利用できる。システムロールへ抽出指示は入れない。

チャットはsnapshotに同梱されたGemma 4正式テンプレートを使う。reasoningは既定で有効であり、推論コマンドの `--thinking` と `--no-thinking` で切り替えられる。vLLMのGemma 4 reasoning parserにより自由形式の思考が終了した後だけJSON Schema制約を適用し、思考と最終JSONを分離する。生成時にはvLLMが対応する型、必須キー、列長、値域、列挙値、文字列長、追加キー禁止を構造化出力で強制する。vLLMが未対応の `uniqueItems` は、該当する整数コード列を初出順のまま重複除去してから完全版スキーマで検証する。この正規化を行ったパスは各JSONL行の `normalized_unique_items` に記録するため、重複だけを理由に症例がエラーJSONLへ送られることはない。正本JSONLの `prediction` は完全版スキーマに準拠する。思考本文は正本JSONLの `reasoning`、完全な生成文字列は `raw_output` に保存する。生成パラメータはsnapshotの `generation_config.json` をvLLMの `generation_config="auto"` で読み込む。現在の公式設定は温度1.0、top-p 0.95、top-k 64、サンプリング有効であり、実際に解決された値を各JSONL行と指標JSONに保存する。

推論環境はvLLM 0.29.0とTransformers 5.17.0を明示的に固定する。構成根拠は [Gemma 4 12Bモデルページ](https://huggingface.co/google/gemma-4-12B-it)、[vLLM 0.29.0 の対応モデル一覧](https://docs.vllm.ai/en/v0.29.0/models/supported_models/)、[vLLM Gemma 4 利用手順](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)、[Modal GPU 指定](https://modal.com/docs/guide/gpu) を参照する。

## OpenAI API

`gpt-5.6-luna`を使う場合はModalやGPUを起動せず、Responses APIへローカルから要求する。APIキーはリポジトリへ保存せず、実行するシェルの環境変数へ設定する。

```bash
export OPENAI_API_KEY="..."

uv run python -m rsna2026.weak_label_openai \
  --input reports.csv \
  --report-col report \
  --output outputs/raw_predictions_gpt56_luna.jsonl \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --max-output-tokens 4096 \
  --workers 8 \
  --resume
```

reasoningを無効化するときは`--reasoning-effort none`を使う。有効時は`low`、`medium`、`high`、`xhigh`、`max`から選ぶ。推論設定が異なる結果は同じJSONLへ再開追記できないため、比較時は出力先を分ける。

既存のエラー記録に含まれる症例だけを再試行する場合は、次を使う。エラー記録の`study_id`だけを入力CSVから選び、正本JSONLに既に存在するIDは除外して追記する。既存行と推論設定が異なる場合は中止する。元のエラー記録は監査用に変更せず、再試行でも失敗した症例は既定で`raw_predictions_gpt56_luna.errors.retry.jsonl`へ記録する。

```bash
uv run python -m rsna2026.weak_label_openai \
  --input reports.csv \
  --report-col report \
  --output outputs/raw_predictions_gpt56_luna.jsonl \
  --retry-errors-from outputs/raw_predictions_gpt56_luna.errors.jsonl \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --max-output-tokens 4096 \
  --workers 8 \
  --resume
```

対象件数だけをAPI要求なしで確認するときは、同じコマンドへ`--dry-run`を加える。再試行の集計は`raw_predictions_gpt56_luna.retry.metrics.json`へ書く。`--error-output`を明示する場合も、元のエラー記録とは別のパスを指定する。

API要求には`strict: true`の構造化出力を指定する。OpenAIの対応スキーマに合わせ、API上の各所見は`m,c,v,a,t,k,r,e`を持つオブジェクトとして生成し、保存前に既存の固定長配列表現へ変換する。OpenAI用スキーマ第2版では、`maxLength`ではなく対応済みの`pattern`を使い、証拠文1件を160文字以下に制約する。`uniqueItems`対象は初出順で重複除去し、完全版スキーマで再検証する。

固定抽出指示はユーザーロールの独立した入力ブロックに置き、その末尾へ明示的なプロンプトキャッシュ境界を設定する。症例固有のStudyInstanceUIDとレポートは次のユーザーメッセージへ置く。キャッシュは明示モード、保持時間は30分で、各行の`usage.cached_input_tokens`と`usage.cache_write_tokens`、集計指標JSONの同名項目から命中状況を確認できる。API応答の保存は`store=false`とする。

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

## 確率較正

三値の`negative=0.1`、`uncertain=0.5`、`positive=0.9`を一律に変更しても、
順位が変わらない限りROC AUCは変化しない。較正時はmentionとcompetition thresholdを
分離し、明示的陰性、未記載、閾値未満、equivocal、情報不足、閾値付近、閾値超過の
順序を使う。さらにcertaintyとseverityを同一カテゴリ内のtie-breakへだけ使用する。

58件の正解ラベルを使う場合は、対象ごとの`(mention, threshold)`陽性率をそのまま使わず、
上記規則を5症例分の事前分布として残す縮約較正を行う。

```bash
uv run python -m rsna2026.weak_label_calibration \
  --input outputs/raw_predictions_gpt56_luna.jsonl \
  --gold-labels data/rsna-knee-abnormality-detection/train.csv \
  --calibration-prior-strength 5 \
  --output outputs/weak_labels_gpt56_luna_calibrated.csv
```

この較正に使った58件上のAUCは、独立な検証値ではない。必ず症例単位の交差検証値も
併記し、confidence weightやmaskの変更でAUCが改善したとは解釈しない。これらは損失への
寄与を制御する値であり、予測順位を変えないためである。

Synovitisが未記載の症例に限り、Effusion抽出値で未記載群内を順位付けする場合は
`--impute-synovitis-from-effusion`を指定する。明示されたSynovitis所見は上書きせず、
Synovitisのconfidence weightと症例maskも変更しない。58件では直接抽出のSynovitis AUC
`0.6726`に対し、この補完は`0.7652`だった。ただし標本数が小さいため、単独の58件評価を
独立な性能保証として扱わない。
