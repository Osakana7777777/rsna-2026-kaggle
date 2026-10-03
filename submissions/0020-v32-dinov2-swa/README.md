# 提出0020: DINOv2・SWA・EMAなし

## 基本情報

| 項目 | 内容 |
|---|---|
| 提出連番 | 0020 |
| Kaggle版（推論） | 32 |
| 学習ノートブック版 | 46 |
| 提出識別子 | 56795849 |
| 提出日時（日本時間） | 2026-10-03 19:34:43 |
| 説明 | DINOv2・SWA15〜20世代・EMAなし・ランキング損失0.05 |
| 公開スコア | 0.895 |
| 非公開スコア | 未公開 |
| 状態 | 完了 |

## 目的

提出0019のDINOv2・ランキング損失0.05にSWAを追加し、後半世代の平均重みの効果を確認する。EMAは使用しない。後半の学習率も変更するため、重み平均だけの効果ではない。基準設定 `configs/dinov2-ranking.toml` と比較設定 `configs/dinov2-ranking-swa.toml` の差はSWA設定のみ。

## 使用データ

全4,407検査・24,371系列、正式ラベル58検査。学習表は `train.csv`、`train_series.csv`。弱教師は `stevenleehans/rsna-knee-llm-report-labels` の `llm_labels_v4_blend.csv`、4,407行、最大学習対象4,345件。弱教師SHA-256は `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`。

推論には `test.csv`、`test_series.csv` と提出見本を使用する。事前学習重みは `metaresearch/dinov2/PyTorch/small/1`。検査・系列選択記録と永続画像キャッシュのハッシュは末尾の確定設定に記載する。

## 前処理

`dicom-physical-fov-window-orientation-uint8-v5`。患者座標で内側を画像右へ正規化し、画像中央130 mmの物理視野を切り出す。1・99百分位正規化、MONOCHROME1反転、224画素への双線形補間。六撮像枠ごとに最大一系列、各系列最大十窓、各窓は物理的に隣接する三枚を使う。欠損撮像枠はマスクする。

4個の永続画像キャッシュを学習前に検証し、665,954枚の復号成功、失敗0枚を確認した。学習時の回転、移動・拡縮、明度・対比、ガンマ、雑音、ぼかしは2.5次元窓内で同期する。数値は末尾の確定設定に記載する。

## モデル

DINOv2-small、末尾六ブロックを学習。三枚を三チャネルとして入力し、分類トークンとパッチ平均を統合する。撮像枠埋め込み・物理位置特徴を加えた256次元特徴へ、8ヘッド・2層の自己注意を適用する。十二所見別の注意集約で出力。脱落率0.25、勾配チェックポイント無効。総パラメータ22,791,564、学習対象11,387,148。

## 学習

報告文の重複群を維持した四分割。外側乱数種42、内側1042、内側検証割合0.20、各分割の学習種42〜45。Tesla T4二枚、NCCLによるデータ並列。各GPU四検査、蓄積一回、実効バッチ八。AdamW、特徴抽出部の最大学習率0.000016、出力部0.002、重み減衰0.02。全分割で20世代を完走し、時間制限到達なし。学習経過は約6.18時間。

正解セルの重みは3.0、弱教師セルは `2 × |p − 0.5|`。重み付き二値交差エントロピーにペアワイズ・ロジスティックランキング損失の0.05倍を加える。ランキングは同じ所見の `p > 0.5` と `p < 0.5` を組にし、GPU間で勾配付き予測共有を行う。ペア重みはセル重みの積。欠損・重み零・p=0.5を除外する。有効所見間で平均し、ペアがなければ零。所見別陽性重みは重み付き陰性量／陽性量（下限一）を0.25〜8に制限する。

1〜14世代はOneCycleLR、立ち上がり比率0.15。15世代開始時にSWALRへ切り替え、一周期更新を停止。二世代かけて学習率を特徴抽出部0.000004、出力部0.0005へ移し、その後固定する。15〜20世代の各終了時の学習モデルを等重みで平均する。初回は15世代の重みをコピーし、16世代以降は平均を更新する。PyTorchの `AveragedModel` と `SWALR` を使用する。EMAなし。DINOv2にバッチ正規化はなく、その統計再計算は不要。

開始前は通常モデル、開始後はその時点までの平均モデルを内側弱ラベルで評価し、所見平均AUCが最大の候補を保存する。外側正式ラベルは選択に使用しない。四分割とも20世代が選ばれ、15〜20世代の六個の重みを平均したモデルが採用された。

## 検証結果

| 分割 | 学習数（正解数） | 内側検証数 | 外側正解数 | 最良世代 | 学習損失 | 内側AUC | 平均数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2654（43） | 651 | 15 | 20 | 0.628074 | 0.844614 | 6 |
| 1 | 2653（43） | 653 | 15 | 20 | 0.718199 | 0.817734 | 6 |
| 2 | 2654（44） | 653 | 14 | 20 | 0.626484 | 0.854437 | 6 |
| 3 | 2651（44） | 652 | 14 | 20 | 0.626943 | 0.843301 | 6 |

正式58検査の分割外平均AUCは **0.85519388**。5,000回の検査単位再標本化による95%区間は 0.820637〜0.886160。提出0019の0.85148826より約0.00371高い。提出0018の0.85584とは近い値。

外側割当SHA-256：`fb14616fde96f5bce78390a052d5af36f2fdffeb4d02013b8d7e67bab200fe18`。
内側割当SHA-256：`8367555064b260cb96d3cb1070be0b34df4b132969a93ae2e4eebec1fce62ac1`。

| 所見 | 分割外AUC |
|---|---:|
| ACL | 0.822304 |
| MCL | 0.820862 |
| Medial Meniscus | 0.830529 |
| Lateral Meniscus | 0.811180 |
| Medial OA | 0.939535 |
| Lateral OA | 0.756286 |
| PF OA | 0.797941 |
| Effusion | 0.920497 |
| Synovitis | 0.786141 |
| Baker's | 0.989130 |
| Contusion | 0.937922 |
| Fracture | 0.850000 |

## 推論と提出ファイル

学習版46の `ensemble-manifest.json` が指定する四個のSWAモデルを推論版32で読み込む。モデル構成の一致を確認し、各所見のテスト予測をモデルごとに百分位順位へ変換して平均する。提出見本の行・列順へ整列する。欠損予測は0.5で補い、値を0〜1へ制限する。

`outputs/submission.csv` を生成し、コード競技用に `/kaggle/working/submission.csv` に配置する。推論版32の構成が学習版46の構成と完全一致することを確認した。公開テストは三検査・十二所見。識別子の一意性、列順、有限値、0〜1の範囲、直下と成果物内CSVの完全一致を確認した。提出CSVのSHA-256は `a9cf659f6a1ffebd406c425042434553fb635ff58557d6832e5986e83eed988a`。推論成果物は `outputs/dinov2-swa-inference-v32/` に取得した。

## 実行・提出基盤

学習：`osakanateishoku/rsna-2026-training` 版46。推論：`osakanateishoku/rsna-2026-workflow` 版32。インターネット無効。学習成果物は `outputs/train/outputs/`、版を指定した検証記録は `outputs/dinov2-swa-v46/outputs/` に取得。`resolved-config.json` とローカルのSWA設定の完全一致、全四モデルのSHA-256を確認した。使用中のCLIは `whoami` 非対応のため、認証を要する私有ノートブック状態取得と完了済み提出履歴で認証・参加を確認した。

```bash
./scripts/verify.sh
./scripts/push.sh train configs/dinov2-ranking-swa.toml
./scripts/output.sh train
./scripts/push.sh submission configs/dinov2-ranking-swa.toml
kaggle competitions submit rsna-knee-abnormality-detection \
  -k osakanateishoku/rsna-2026-workflow \
  -v 32 -f submission.csv \
  -m "DINOv2・SWA15〜20世代・EMAなし・ランキング損失0.05"
```

各分割の保存重みSHA-256：

- 分割0：`3f30e7f22017d01c7a375b7c10483711b1f36d8f22838fba504ecabd1314226c`。
- 分割1：`3bb4b7b61b41f77d2f1bb54128c40f6a5601f143f3563d8d7df8cee61efbef05`。
- 分割2：`8295f9b6e5bf477b140c720552dc6ff89af35edc8564467074566a2a1348dd86`。
- 分割3：`aff5885b91a585bc4a3fd15ac737481857866dd3439a78597068436c79e3f5eb`。

SWA専用試験14件と構成検証は成功。コミット対象の七ファイルと既存の追跡済みファイルだけを一時領域に配置し、全試験230件成功・一件省略を確認した。学習用・推論用の送信用ファイル作成と構成検証も成功した。作業領域全体を確認した際のCoAtNet設定に関する二件の不一致は `notes/1003-swa-plan.md` に記録したが、それらの別実験の変更は今回のコミットに含めない。

## 結果

2026年10月3日19:34:43（日本時間）に提出識別子56795849で受け付けられた。公開スコアは **0.895** で完了した。SWAなしの提出0019（0.899）より0.004低い一方、CVは0.85148826から0.85519388へ約0.00371上がった。公開スコアとCVの方向は一致していないため、この一回だけでSWAの採否を確定しない。

## 既知の制約

正式ラベル58検査のため評価区間が広い。SWAと後半の学習率変更を同時に行っており、それぞれの寄与は切り分けられない。全所見で一様に改善したわけではない。公開スコアだけで採用を確定しない。

## 次の改善候補

1. 同じ検査の分割外予測でSWAなしとの差の再標本化区間と所見別の差を確認する。
2. 平均開始世代と後半の固定学習率を、内側検証を使って比較する。
3. 初期値を変えた再実験と、既存モデルとの予測混合を検証する。

## 確定設定

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
# Average each completed epoch from 15 through 20; no EMA.
swa_enabled = true
swa_start_epoch = 15
swa_anneal_epochs = 2
swa_backbone_learning_rate = 0.000004
swa_head_learning_rate = 0.0005
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
