# 提出0021: RadImageNet ResNet50全固定・Transformer MIL

## 基本情報

| 項目 | 内容 |
|---|---|
| 提出連番 | 0021 |
| Kaggle版（推論） | 33 |
| 学習ノートブック版 | 1 |
| 提出識別子 | 56812531 |
| 提出日時（日本時間） | 2026-10-04 11:59:45 |
| 説明 | RadImageNet ResNet50全固定・Transformer MIL・4分割・固定学習率0.0002 |
| 公開スコア | 0.889 |
| 非公開スコア | 未公開 |
| 状態 | 完了 |

## 目的

RadImageNet事前学習ResNet50を全固定し、既存のTransformer MIL集約部のみ学習したモデルを正式提出する。公開学習コードの集約部学習率0.0002、実効バッチ48、最大24世代、改善幅0.0002・猶予5世代の早期終了を参考にした。入力と集約部は元のRadImageNet設定を維持した。入力・教師・重み減衰・集約部構造が公開コードと異なるため、公開モデルの完全再現ではない。

## 使用データ

全4,407検査・24,371系列、正式ラベル58検査。学習表は `train.csv`、`train_series.csv`。弱教師は私有データ `osakanateishoku/rsna-2026-gpt56-luna-weak-labels` の `weak_labels_gpt56_luna_calibrated_synovitis_imputed.csv`、4,407行。SHA-256は `b50a4ccb89e2232595d249d0c428673e3aaaf737a419054f9097ca446a278eee`。正式58検査と照合した教師の平均AUCは0.88845557で、受入基準0.88を満たした。

全検査で報告文の重複群を維持した四分割を設計し、その後、CSVの信頼度・除外指定を反映して重みがすべて零の症例を学習と内側検証から除いた。実際の学習・検証数は下表。元設定の `max_studies=3303` は単一分割用の選択設定で、この交差検証の症例数上限ではない。

推論には `test.csv`、`test_series.csv` と提出見本を使う。事前学習重みは `osakanateishoku/radimagenet-resnet50` の `ResNet50.pt`。競技画像はローカルへ取得していない。

## 前処理

`dicom-physical-fov-window-orientation-uint8-v5`。患者座標で内側を画像右へ正規化し、中央130 mmの物理視野を切り出す。1・99百分位正規化、MONOCHROME1反転、224画素への補間。六撮像枠ごとに最大一系列、各系列十窓、隣接三枚を三チャネルにする。欠損枠はマスクする。ResNetへの入力は−1〜1へ正規化する。

学習では4個の永続画像キャッシュを使用。665,954枚、復号失敗0枚を起動時に検証済み。回転・移動・拡縮・明度・対比・ガンマ・雑音・ぼかしを窓内で同期する。数値とキャッシュのハッシュは末尾の確定設定に記載する。推論時はテスト画像を処理し、学習画像キャッシュは使わない。

## モデル

RadImageNet ResNet50の分類層を除き、全重みとバッチ正規化統計を固定。各窓から2048次元特徴を得る。256次元への投影、物理位置特徴、撮像枠埋め込みを加え、8ヘッド・2層の自己注意と残差加算を適用する。所見別注意で全窓を集約して十二所見を出力する。集約部は `TransformerTargetAttentionHead`。標準TransformerEncoderのような独立したフィードフォワード層はない。脱落率0.25。総パラメータ24,573,260、学習対象1,065,228。

## 学習

Tesla T4一枚、四分割を順番に学習。乱数種42〜45、外側分割42、内側1042、内側検証割合0.20。AdamW、学習対象の固定学習率0.0002、重み減衰0.02、EMAとSWAなし。設定にある特徴抽出部学習率0.000016は、全固定のため更新に使われない。

各処理四検査、勾配累積12回、実効バッチ48。各四検査の重み付き損失を平均して累積するため、48検査を一度に処理した重み付き平均とは完全には同じ勾配にならない。所見別信頼度にCSVの症例マスクを乗じ、正式ラベルは重み14を優先する。損失は所見別陽性重み付き二値交差エントロピー。陽性重みは重み付き陰性量／陽性量（分母下限一）を0.25〜8に制限する。ランキング損失なし。

内側弱ラベル検証の所見平均AUCが保存済み最良値を0.0002超上回ったとき更新する。改善なしが5世代続いたら早期終了し、最良重みを復元する。外側正式ラベルは世代選択に使わない。最大全24世代、全分割合計の時間上限12時間。時間上限到達なし。

## 検証結果

| 分割 | 学習数（正式数） | 内側検証数 | 外側正式数 | 実行世代 | 最良世代 | 最良世代の学習損失 | 内側AUC | 終了理由 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2017（43） | 508 | 15 | 24 | 20 | 0.723213 | 0.864300 | 上限到達 |
| 1 | 2015（43） | 519 | 15 | 17 | 12 | 0.748288 | 0.880820 | 早期終了 |
| 2 | 2017（44） | 499 | 14 | 13 | 8 | 0.796953 | 0.854003 | 早期終了 |
| 3 | 2003（44） | 511 | 14 | 14 | 9 | 0.781669 | 0.848299 | 早期終了 |

正式58検査の分割外平均AUCは **0.84949348**。検査単位5,000回の再標本化95%区間は0.808312〜0.888159。学習全体は約4.13時間。

外側割当SHA-256：`988e114372be512c625ef972af829785c71f891e7cbe6a3e25fb8bd34da27e3a`。
内側割当SHA-256：`0b33cc79b43071640a15f4346b50cf6102c0c88b316836decaf7609733ed1ae4`。

| 所見 | 分割外AUC |
|---|---:|
| ACL | 0.862745 |
| MCL | 0.861678 |
| Medial Meniscus | 0.875000 |
| Lateral Meniscus | 0.762733 |
| Medial OA | 0.959690 |
| Lateral OA | 0.849130 |
| PF OA | 0.813385 |
| Effusion | 0.927950 |
| Synovitis | 0.729988 |
| Baker's | 0.871377 |
| Contusion | 0.867746 |
| Fracture | 0.812500 |

## 推論と提出ファイル

学習版1の `ensemble-manifest.json` が指定する四モデルを推論版33で使用する。各重みのSHA-256とモデル構成をKaggle推論内で検証し、所見ごとに各モデル予測を百分位順位へ変換して平均する。提出見本の行・列順へ整列し、欠損予測は0.5、値を0〜1へ制限する。学習や画像拡張は行わない。提出ファイルは `outputs/submission.csv` とコード競技用の `/kaggle/working/submission.csv`。

推論版33は正常終了し、公開テスト3検査・十二所見を約29.87秒で処理した。識別子の一意性、列順、有限値、0〜1の範囲、直下と成果物内CSVの完全一致を確認した。学習版1の構成と送信済み推論設定、および学習の集約記録と推論要約の集約記録が完全一致した。提出CSVのSHA-256は `f468c8748cb9e873ff5d3ccda7abf62b25ac4ac4ae528f40b2390523d0992c7e`。推論成果物と検証記録は `outputs/radimagenet-inference-v33/` に保存。

## 実行・提出基盤

学習：[RadImageNet版1](https://www.kaggle.com/code/osakanateishoku/rsna-2026-radimagenet-resnet50)。推論：[Workflow](https://www.kaggle.com/code/osakanateishoku/rsna-2026-workflow)、版33。両方インターネット無効。推論には学習成果物とRadImageNet重みだけを接続する。`scripts/common.sh` の `radimagenet-submission` と専用メタデータで再実行できる。

CLIは `whoami` 非対応のため、私有ノートブック状態取得と提出履歴で認証・競技アクセスを確認した。学習の確定設定とローカル設定は全項目一致。四分割の重みハッシュは保存記録と集約記録が一致することを確認した。学習成果物の要約は `outputs/radimagenet/`、送信した推論ソースと設定ハッシュは `outputs/radimagenet-inference-launch/`。

```bash
./scripts/verify.sh
./scripts/prepare.sh train
./scripts/prepare.sh submission
./scripts/push.sh radimagenet-submission
kaggle competitions submit rsna-knee-abnormality-detection \
  -k osakanateishoku/rsna-2026-workflow \
  -v 33 -f submission.csv \
  -m "RadImageNet ResNet50全固定・Transformer MIL・4分割・固定学習率0.0002"
```

正式提出コマンドは成功し、提出識別子56812531で受付済み。採点結果は `SubmissionStatus.COMPLETE`、公開スコア0.889。保存重みのSHA-256：

- 分割0：`c3df34bdcd4ef52a1dfbb5ed7d9264c9fc57326cd0988f993fb3b158ff30ff8e`。
- 分割1：`5f6cb11209ed12b37f57859b15f3056a61c6b8fe2a86968d69563a2063e482f5`。
- 分割2：`2c7420ab62c36d0f25d302aaca85d227d2e94861251ca38443061ad4f053cf1d`。
- 分割3：`58c88b8a79952b7559e37f2fa5a2f64e0e19fbc5d3688712123272cd86818409`。

`./scripts/verify.sh` と通常学習・通常推論・ResNet推論の送信内容照合は成功。関連試験46件が成功。コミット対象14ファイルと既存の追跡済みファイルだけを一時領域に配置し、全試験は合計238件成功・1件省略を確認した。分散試験1件はサンドボックスのローカル通信制限で失敗したため、その1件を制限外で再実行して成功した。コミット対象の通常学習・通常推論・ResNet推論の生成ファイルで、実行方式・埋め込まれた設定とソースの一致も確認した。

## 結果

2026年10月4日11:59:45（日本時間）に提出識別子56812531で受け付けられた。公開スコアは **0.889** で完了。直前のDINOv2・SWA提出0020（0.895）より0.006低い。教師や損失重みなども異なるため、ResNet全固定だけの効果とは解釈しない。

## 既知の制約

正式ラベル58検査で評価区間が広い。過去のDINOv2提出とは事前学習重みだけでなく、教師・信頼度・除外指定・正式ラベル重み・学習率も異なり、特徴抽出部だけの比較ではない。公開コードとの一致は学習率・実効バッチ・世代上限・早期終了条件を中心とし、入力・集約部・重み減衰・勾配の平均方法は異なる。学習成果物の接続はKaggleで最新の完了版を参照するため、学習ノートブックを更新した後は版1を使ったことを別途確認する必要がある。

## 次の改善候補

1. 同じ正式検査でDINOv2と所見別予測の相関を確認し、混合の効果を検証する。
2. 教師と損失重みをそろえて特徴抽出部の効果を比較する。
3. 内側検証で集約部学習率と一部のResNet層の学習を比較する。

## 確定設定

```toml
[competition]
input_slug = "rsna-knee-abnormality-detection"

[paths]
input_root = "/kaggle/input"
working_root = "/kaggle/working"
output_dir = "/kaggle/working/outputs"

[run]
seed = 42

[data]
image_size = 224
physical_fov_mm = 130.0
window_count = 10
window_size = 3
max_series_per_slot = 1
dicom_cache_dir = "/tmp/rsna2026-dicom-cache"
series_classification = "competition-series-table-v1"
study_records_cache_path = "/kaggle/input/rsna-2026-slice-selection-cache/slice-selection.json.xz"
study_records_cache_sha256 = "5d639d8a41b4001b0eed8398e6a3a2cdc680eb06cb025a3a24695cc6cc26eb01"
study_records_cache_uids_sha256 = "72bb18659cba9fc84f4f363c2585f1a99f84e6f03971e4d8313813ba647aa81e"
slots = [
  "sagittal-fluid",
  "coronal-fluid",
  "axial-fluid",
  "sagittal-structural",
  "coronal-structural",
  "axial-structural",
]

[image_cache]
enabled = true
manifest_paths = [
  "/kaggle/input/rsna-2026-preprocess-shard-0/outputs/image-cache/cache-manifest.json",
  "/kaggle/input/rsna-2026-preprocess-shard-1/outputs/image-cache/cache-manifest.json",
  "/kaggle/input/rsna-2026-preprocess-shard-2/outputs/image-cache/cache-manifest.json",
  "/kaggle/input/rsna-2026-preprocess-shard-3/outputs/image-cache/cache-manifest.json",
]
manifest_sha256 = [
  "432466d2ceabeefad9b8c77169fdd357d92def685dc637a9d7360a6c63d301f4",
  "27c0bef3bd2c7614bf1aba0d8bd63831c5744c997a13e34776732568846da737",
  "fe238119835aefd358a97efd9110f6f177630d7d4bcc3fab3e48aa17956f9e79",
  "e89b0cc5a6c1ba7ad9bd68bae46ff32dc1df0ac90d3a79557d6ffe932763c481",
]
expected_studies = 4407
expected_series = 24371
expected_images = 665954
expected_failed_images = 0
expected_selected_series = 21334

[image_cache.expected_missing_by_slot]
sagittal-fluid = 257
coronal-fluid = 159
axial-fluid = 0
sagittal-structural = 141
coronal-structural = 1001
axial-structural = 3550

[augmentation]
enabled = true
rotation_degrees = 8.0
rotation_probability = 0.5
translation_fraction = 0.05
scale_min = 0.92
scale_max = 1.08
affine_probability = 0.5
brightness_limit = 0.10
contrast_limit = 0.10
brightness_contrast_probability = 0.4
gamma_min = 0.85
gamma_max = 1.15
gamma_probability = 0.3
noise_std_max = 0.02
noise_probability = 0.2
blur_kernel_sizes = [3, 5]
blur_sigma_min = 0.1
blur_sigma_max = 1.0
blur_probability = 0.1

[training]
epochs = 24
# Four studies per forward pass, twelve accumulation steps: effective batch 48.
batch_size_per_device = 4
gradient_accumulation_steps = 12
inference_batch_size = 1
num_workers = 2
distributed = false
expected_gpu_count = 1
backbone_learning_rate = 0.000016
head_learning_rate = 0.0002
scheduler = "constant"
early_stopping_patience = 5
early_stopping_min_delta = 0.0002
weight_decay = 0.02
one_cycle_warmup_fraction = 0.15
max_runtime_hours = 12.0
validation_fraction = 0.2
max_train_studies = 0

[cross_validation]
enabled = true
folds = 4
outer_seed = 42
inner_seed = 1042
inner_validation_fraction = 0.20
gold_loss_weight = 14.0
bootstrap_resamples = 5000
test_ensemble = "per-target-percentile-rank-mean"
split_algorithm = "report-group-multilabel-v1"

[pseudo_labels]
use_explicit_supervision = true
enabled = true
dataset = "osakanateishoku/rsna-2026-gpt56-luna-weak-labels"
file_name = "weak_labels_gpt56_luna_calibrated_synovitis_imputed.csv"
source_path = "/kaggle/input/rsna-2026-gpt56-luna-weak-labels/weak_labels_gpt56_luna_calibrated_synovitis_imputed.csv"
sha256 = "b50a4ccb89e2232595d249d0c428673e3aaaf737a419054f9097ca446a278eee"
expected_rows = 4407
max_studies = 3303
expected_validation_studies = 58
minimum_source_macro_auc = 0.88

[model]
name = "radimagenet-resnet50"
variant = "resnet50"
weights_file = "ResNet50.pt"
normalization = "minus-one-to-one"
input_representation = "2.5d-slice-channels"
unfreeze_last = 0
gradient_checkpointing = false
pooling = "global-average"
aggregation = "transformer-target-attention"
attention_hidden_size = 256
transformer_layers = 2
transformer_heads = 8
dropout = 0.25

[inference]
ensemble_manifest_path = "/kaggle/input/rsna-2026-radimagenet-resnet50/outputs/ensemble-manifest.json"

```
