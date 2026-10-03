# 提出0019: DINOv2・ランキング損失追加

## 基本情報

| 項目 | 内容 |
|---|---|
| 提出連番 | 0019 |
| Kaggle版（推論） | 31 |
| 学習ノートブック版 | 45 |
| 提出識別子 | 56787504 |
| 提出日時（日本時間） | 2026-10-03 11:24:47 |
| 説明 | DINOv2・線形重み・4分割・ランキング損失0.05 |
| 公開スコア | 0.899 |
| 非公開スコア | 未公開 |
| 状態 | 完了 |

## 目的

DINOv2 の既存の重み付き二値交差エントロピーに、ペアワイズ・ロジスティックランキング損失の0.05倍を加える効果を確認する。線形重みを採用し、GPU2枚のデータ並列で4分割を学習した。提出0018と外側・内側の割当、モデル構成、主要学習率は同じだが、バッチ構成と画像キャッシュの復号失敗数も異なるため、ランキング損失だけの厳密な比較ではない。

## 使用データ

- 全4,407検査、24,371系列。正解ラベルは58検査、外側保留数は15・15・14・14。
- 学習表は `train.csv`、`train_series.csv`。推論は `test.csv`、`test_series.csv` と提出見本を使用する。公開テストは3検査。
- 弱教師は `stevenleehans/rsna-knee-llm-report-labels` の `llm_labels_v4_blend.csv`、4,407行。
- 弱教師SHA-256：`eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`。
- 弱教師の正式58検査での平均AUCは0.892707。受け入れ基準0.88を満たした。
- 事前学習重みは `metaresearch/dinov2/PyTorch/small/1`。
- 選択記録SHA-256：`5d639d8a41b4001b0eed8398e6a3a2cdc680eb06cb025a3a24695cc6cc26eb01`。
- 全検査識別子集合SHA-256：`72bb18659cba9fc84f4f363c2585f1a99f84e6f03971e4d8313813ba647aa81e`。

## 前処理

`dicom-physical-fov-window-orientation-uint8-v5`。患者座標によって内側を画像右に正規化し、中央130 mmを切り出す。1・99百分位による正規化、MONOCHROME1反転、224画素への双線形補間を行う。6撮像枠ごとに最大1系列、各系列最大10窓、各窓は物理座標上で隣接する3枚を使用する。

4つの永続画像キャッシュで665,954枚すべてを復号し、失敗は0枚。キャッシュの検査数・系列数・画像数・ハッシュを学習前に検証した。欠損撮像枠はマスクする。画像拡張の全数値と各キャッシュのハッシュは末尾の確定設定に記録した。

## モデル

DINOv2-smallの末尾6ブロックを学習し、分類トークンとパッチ平均を統合する。3枚を入力の3チャネルとする2.5次元表現。特徴、撮像枠埋め込み、物理位置特徴を256次元へ射影し、8ヘッド・2層の自己注意を適用して、12所見別の注意集約で出力する。脱落率0.25、勾配チェックポイントは無効。

総パラメータ22,791,564、学習対象11,387,148。

## 学習

報告文の重複群を維持した4分割。外側乱数種42、内側1042、内側検証割合0.20。学習乱数種42～45。各分割20世代を完走し、時間制限到達はない。Tesla T4を2枚、NCCLによるデータ並列、各GPU4検査、勾配蓄積1回、実効バッチ8。AdamW、符号器学習率0.000016、集約・出力部0.002、重み減衰0.02、一周期方式の立ち上がり比率0.15。移動平均モデルは使用しない。

既存損失は、正解セル重み3.0、弱教師セル重み `2 × |p − 0.5|` を用いた二値交差エントロピー。所見ごとの陽性重みは、重み付き陰性量を重み付き陽性量（下限1）で割り、0.25～8に制限して計算する。セル損失の重み付き和をセル重みの和（下限1）で割る。

追加損失は同一所見で `p > 0.5` の症例と `p < 0.5` の症例を組にし、`softplus(−(陽性の出力値 − 陰性の出力値))` を計算する。ペア重みは両セル重みの積。所見ごとに重み付き平均を取り、有効なペアのある所見間で平均する。欠損、重み0、`p = 0.5` は除外し、有効なペアがなければ0。GPU間で勾配付きの予測共有を行い、全体8症例のペアを使用する。最終損失は既存損失にこの損失の0.05倍を加えたもの。

最良世代は内側弱ラベルの所見平均AUCで選ぶ。外側の正解ラベルは選択に使用しない。

## 検証結果

| 外側分割 | 学習数（正解数） | 内側検証数 | 最良世代 | その世代の学習損失 | 内側AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | 2654（43） | 651 | 17 | 0.627325 | 0.845639 |
| 1 | 2653（43） | 653 | 19 | 0.626662 | 0.832412 |
| 2 | 2654（44） | 653 | 17 | 0.657862 | 0.858812 |
| 3 | 2651（44） | 652 | 17 | 0.649260 | 0.836211 |

正式58検査の分割外平均AUCは **0.85148826**。5,000回の再標本化による95%区間は 0.814269～0.884034。

外側割当SHA-256：`fb14616fde96f5bce78390a052d5af36f2fdffeb4d02013b8d7e67bab200fe18`。
内側割当SHA-256：`8367555064b260cb96d3cb1070be0b34df4b132969a93ae2e4eebec1fce62ac1`。

| 所見 | 分割外AUC |
|---|---:|
| ACL | 0.843137 |
| MCL | 0.877551 |
| Medial Meniscus | 0.893029 |
| Lateral Meniscus | 0.655901 |
| Medial OA | 0.931783 |
| Lateral OA | 0.746615 |
| PF OA | 0.779923 |
| Effusion | 0.954037 |
| Synovitis | 0.746714 |
| Baker's | 0.945652 |
| Contusion | 0.936572 |
| Fracture | 0.906944 |

## 推論と提出ファイル

学習版45の `ensemble-manifest.json` が指定する4モデルを推論版31で読み込む。モデル構成の一致を確認し、所見ごとに各モデルのテスト予測を百分位順位へ変換して平均する。提出見本に合わせた行・列順で `outputs/submission.csv` を書き、コード競技用に `/kaggle/working/submission.csv` へ配置する。

推論成果物の4モデル構成と学習版45の構成が一致した。提出ファイルは3検査・12所見で、検査識別子の一意性、列順、有限値、0～1の範囲、作業領域直下と成果物内のファイル一致を確認した。欠損予測は0.5で補い、0～1へ制限する実装。提出CSVのSHA-256は `54a11281962b764ef8d01b8b90efdba0cc964f05e312625c7c9067ad3d3fc42c`。

## 実行・提出基盤

学習ノートブック：`osakanateishoku/rsna-2026-training` 版45。推論ノートブック：`osakanateishoku/rsna-2026-workflow` 版31。インターネット無効。既存のKaggle CLIは `whoami` 非対応のため、認証済みAPIによる私有ノートブック状態および競技提出履歴の取得で認証と参加を確認した。

```bash
./scripts/verify.sh
./scripts/push.sh train configs/dinov2-ranking.toml
./scripts/push.sh submission configs/dinov2-ranking.toml
kaggle competitions submit rsna-knee-abnormality-detection \
  -k osakanateishoku/rsna-2026-workflow \
  -v 31 -f submission.csv \
  -m "DINOv2・線形重み・4分割・ランキング損失0.05"
```

学習成果物は `outputs/dinov2-ranking-v45/outputs/` に取得。`resolved-config.json` とローカル設定の一致を確認した。設定・ランキング損失・分散勾配・生成処理の関連テスト34件は成功。コミット対象だけを一時領域へ取り出し、ランキング損失・分散勾配・既存設定・モデルなどの関連テスト84件が成功した。構文・構成・入力データ持ち出し防止の検証も成功した。

各分割のモデルSHA-256：

- 分割0：`9a91976d38e1587fa1eba06feb656a6acc9ad9f0d334c93cfdb61e0721f3376f`。
- 分割1：`821eed71d4947df3f6d53850e7eace1f9e4409cf9842ab65782e97f5e68eebe4`。
- 分割2：`cbc9ac8aaf36b5d5c92aafb33ffb274ac5746c48477a0cc52a811cc603054e9a`。
- 分割3：`650fbdf50b2ac7b510a9b5a18574385031a4592dd6e719a7831ec372cb3ae45a`。

## 結果

2026年10月3日11:24:47（日本時間）に提出識別子56787504で受け付けられた。公開スコアは **0.899** で完了した。提出0018の0.893より0.006高く、提出0016・0017の0.899・0.897と同等以上だった。一方、分割外AUCは提出0018の0.85584を約0.00435下回った。バッチ構成とキャッシュ状態も異なるため、公開スコアの向上をランキング損失だけの効果とは断定しない。

## 既知の制約

- 正解ラベル58検査での評価であり、再標本化区間が広い。
- ランキングはバッチ内ペアのみ。弱教師の確率を0.5で区切るため、確率の細かな差は順位目標に使わない。
- 有効な陽性・陰性ペアのない所見は追加損失に寄与しない。
- 推論時の百分位順位はテスト症例数に依存し、確率校正の値ではない。
- 提出0018との比較ではバッチ構成とキャッシュの復号状態も異なる。

## 次の改善候補

1. 同一バッチ構成・同一キャッシュ・同一分割で、追加損失の係数0を対照として比較する。
2. 係数0.01・0.05・0.1を比較する。
3. 所見別AUCと有効ペア数を確認し、弱教師に対するペア選択方法を検討する。

## 確定設定

学習成果物の `resolved-config.json` と一致を確認した設定。

```toml
[competition]
# The competition folder name. Both /kaggle/input/<slug> and
# /kaggle/input/competitions/<slug> are supported.
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
# Reuse decoded, windowed, and resized slices across epochs. This ephemeral path
# is not included in the training notebook outputs.
dicom_cache_dir = "/tmp/rsna2026-dicom-cache"
series_classification = "competition-series-table-v1"
# Reuse the pinned selection records and validate their contents before training.
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
epochs = 20
# Two GPUs process four studies each: global effective batch eight.
batch_size_per_device = 4
gradient_accumulation_steps = 1
inference_batch_size = 1
num_workers = 2
distributed = true
expected_gpu_count = 2
distributed_strategy = "data-parallel"
pairwise_ranking_loss_weight = 0.05
backbone_learning_rate = 0.000016
head_learning_rate = 0.002
weight_decay = 0.02
one_cycle_warmup_fraction = 0.15
max_runtime_hours = 12.0
validation_fraction = 0.2
# Set to a positive value for a quicker diagnostic run; 0 uses every labeled study.
max_train_studies = 0

[cross_validation]
# Train all four outer folds.
enabled = true
folds = 4
outer_seed = 42
inner_seed = 1042
inner_validation_fraction = 0.20
gold_loss_weight = 3.0
bootstrap_resamples = 5000
test_ensemble = "per-target-percentile-rank-mean"
split_algorithm = "report-group-multilabel-v1"

[pseudo_labels]
confidence_weight_method = "linear-distance"
enabled = true
dataset = "stevenleehans/rsna-knee-llm-report-labels"
file_name = "llm_labels_v4_blend.csv"
source_path = "/kaggle/input/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
sha256 = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
expected_rows = 4407
max_studies = 4345
expected_validation_studies = 58
minimum_source_macro_auc = 0.88

[model]
name = "dinov2-small"
variant = "small"
input_representation = "2.5d-slice-channels"
unfreeze_last = 6
gradient_checkpointing = false
pooling = "cls-patch-mean"
aggregation = "transformer-target-attention"
attention_hidden_size = 256
transformer_layers = 2
transformer_heads = 8
dropout = 0.25

[inference]
# Training-notebook outputs are mounted below /kaggle/input using the notebook slug.
ensemble_manifest_path = "/kaggle/input/rsna-2026-training/outputs/ensemble-manifest.json"
```
