# RSNA 2026: Codex → Kaggle GPU workflow

Codex edits this repository locally. Training and code-competition inference run
as separate private Kaggle notebooks. The training notebook publishes a verified
checkpoint; the submission notebook mounts that output and never trains.

Competition data is never downloaded to this computer. Kaggle attaches it to the
run as a read-only input below `/kaggle/input`, using `competition_sources` in
`kaggle/kernel-metadata.json`.

## Layout

```text
rsna-2026-kaggle/
├── configs/baseline.toml       # experiment and mount placeholders
├── kaggle/
│   ├── train-kernel-metadata.json # training notebook and inputs
│   ├── kernel-metadata.json    # inference/submission notebook and inputs
│   └── launcher.py             # remote entry point
├── scripts/
│   ├── prepare.sh              # builds train, submission, or EDA payload
│   ├── push.sh                 # uploads train or submission notebook
│   ├── status.sh               # reports the latest run state
│   ├── wait.sh                 # polls until completion or failure
│   ├── output.sh               # retrieves /kaggle/working artifacts
│   └── verify.sh               # checks structure and safety policy
├── submissions/                # detailed Markdown record for every submission
├── src/rsna2026/               # locally edited experiment package
├── outputs/                    # downloaded run artifacts; ignored by Git
└── .kaggle-build/              # generated upload payload; ignored by Git
```

## One-time setup

1. Install `uv`, then install the official Kaggle command-line client:

   ```bash
   uv tool install kaggle
   ```

2. Create an API token on the Kaggle account settings page and configure the
   Kaggle client. Keep the token outside this repository. Confirm authentication:

   ```bash
   kaggle whoami
   ```

3. Accept the competition rules on Kaggle. API authentication alone does not
   grant access to competition data.

4. Configure the account and competition identifiers in both metadata files and
   `configs/baseline.toml` if this repository is copied to another account:

   - training notebook: `kaggle/train-kernel-metadata.json`
   - submission notebook: `kaggle/kernel-metadata.json`
   - mounted competition folder: `configs/baseline.toml`

   The competition slug must also be the mounted competition folder name.
   Both `/kaggle/input/<slug>` and `/kaggle/input/competitions/<slug>` layouts
   are detected. If Kaggle uses a different folder name, set
   `competition.input_slug` to that name while leaving `competition_sources`
   as the official competition slug.

## Train once

From this directory:

```bash
./scripts/verify.sh
./scripts/push.sh train
./scripts/status.sh train
./scripts/wait.sh train
./scripts/output.sh train
```

## Run EDA and download its outputs

The EDA notebook uses the competition mount only, requires no GPU, and saves
tables, charts, and representative DICOM image panels below
`/kaggle/working/outputs/eda`. Run the entire remote workflow with:

```bash
./scripts/run-eda.sh
```

This builds and uploads the private `osakanateishoku/rsna-2026-eda` notebook,
waits for completion, then downloads its artifacts to `outputs/eda/` on this
computer. To run stages separately, use `./scripts/push.sh eda`,
`./scripts/wait.sh eda`, and `./scripts/output.sh eda`.

This uploads `osakanateishoku/rsna-2026-training`. It reads the public soft
labels and competition training images, trains and validates the model, and
writes `best-model.pt` with `checkpoint-manifest.json`. The manifest records the
checkpoint SHA-256 and exact architecture. Training provenance remains in
`run-summary.json`, `label-source-validation.json`, and `slice-selection.json`.
The training notebook does not read the test tables or create a submission file.

## Build the inference notebook

Only after the training notebook completes successfully:

```bash
./scripts/push.sh submission
./scripts/status.sh submission
./scripts/wait.sh submission
./scripts/output.sh submission
```

This uploads `osakanateishoku/rsna-2026-workflow`. Its `kernel_sources` mounts the
saved output of `osakanateishoku/rsna-2026-training`. Before inference, it verifies
the checkpoint filename, SHA-256, target order, image size, and architecture. It
then reads only the test tables and writes `/kaggle/working/submission.csv`.
After creating a new training version, always push a new submission-notebook
version so that its notebook input is pinned to the new training output.

Both stages embed the same source and configuration, but their generated run
modes are fixed to `train` and `inference`. A configuration edit cannot make the
submission notebook enter the training path.

The model classifies series into six plane-and-acquisition slots and retains the
two series with the largest slice counts in each slot. It orders every retained
series on the DICOM patient-coordinate axis, forms adjacent three-slice windows,
and selects up to ten unique windows nearest equally spaced physical centers over
the usable acquisition range. DINOv2 encodes every valid window in one model call.
Target-specific masked attention aggregates windows within each series, series
within each slot, and finally the six slots. Physical position and adjacent-spacing
features accompany each window. Decoded and resized source images are cached once
even when they occur in overlapping windows. Training augmentation is synchronized
across all series and windows in the same slot; validation and inference remain
deterministic. All experiment settings are defined in `configs/baseline.toml`.

The Kaggle image supplies PyTorch, pandas, NumPy, and pydicom. Internet access
remains disabled during the run, and the baseline does not require downloaded
weights.

After a successful inference run, submit its version number:

```bash
kaggle competitions submit rsna-knee-abnormality-detection \
  -k osakanateishoku/rsna-2026-workflow \
  -v VERSION \
  -f submission.csv \
  -m "Multi-view CNN baseline"
```

Every formal submission must also have a detailed record below `submissions/`.
Copy `submissions/TEMPLATE.md`, name its directory with the submission sequence,
Kaggle version, and a short experiment name, then record the exact configuration,
processing, validation, score, limitations, and next steps. Update
`submissions/README.md` after the score is available.

For a complete new training and inference cycle:

```bash
./scripts/push.sh train && ./scripts/wait.sh train && ./scripts/output.sh train
./scripts/push.sh submission && ./scripts/wait.sh submission && ./scripts/output.sh submission
```

## Run training on Runpod

Runpod is an alternative training host; Kaggle remains the submission host. Install
and authenticate the current `runpodctl`, then inspect the GPU IDs currently in
stock and create a Pod with an explicit GPU type and count:

```bash
runpodctl doctor
./scripts/runpod.sh gpus
./scripts/runpod.sh create \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 2 \
  --network-volume-id YOUR_NETWORK_VOLUME_ID
```

Use `--dry-run` to inspect the complete creation command without renting a Pod.
The default is Secure Cloud, two GPUs, a current Runpod PyTorch image, a 50 GB
container disk, and `/workspace` as the persistent-volume mount. A disposable
200 GB Pod volume is created when no network-volume ID is supplied. Pod billing
continues until it is stopped or deleted:

```bash
./scripts/runpod.sh get POD_ID
./scripts/runpod.sh stop POD_ID
./scripts/runpod.sh delete POD_ID
```

The script does not copy competition data. Stage only data whose competition
rules permit use on external compute. The default input layout on the persistent
volume is:

```text
/workspace/input/
├── rsna-knee-abnormality-detection/
│   ├── train.csv
│   ├── train_series.csv
│   └── train_series/
├── rsna-knee-llm-report-labels/
│   └── llm_labels_v4_blend.csv
├── rsna-2026-slice-selection-cache/
│   └── slice-selection.json.xz
└── dinov2-small/
    ├── config.json
    └── model.safetensors (or the equivalent local model-weight file)
```

After connecting to the Pod, place this repository under `/workspace`, install
`uv`, and start training from the repository. The runner creates a uv virtual
environment that can see the PyTorch image's system packages, installs `pydicom`
and `transformers` if needed, validates CUDA, and uses all visible GPUs by default:

```bash
./scripts/runpod-train.sh \
  --input-root /workspace/input \
  --output-dir /workspace/outputs/experiment-name
```

Pass `--gpu-count 1` (or another count no larger than the Pod allocation) to use
a subset of visible GPUs. Pass `--build-selection-cache` if the pinned selection
cache was not staged; this rebuilds it from the staged DICOM headers before
training. Host-specific paths and GPU count are runtime overrides, so
`configs/baseline.toml` remains identical to the Kaggle experiment definition.

## 実験0002: 公開LLM混合ラベル

学習ノートブックは公開データセット
`stevenleehans/rsna-knee-llm-report-labels` の
`llm_labels_v4_blend.csv` を入力として参照します。ファイルのSHA-256、4,407行、
十二所見の列、識別子集合、欠損、値域を学習前に検証し、公開データセットの更新へ
自動追従しません。

正式ラベル58件はすべて検証専用です。同じ正規化報告文を持つ検査を候補から外し、
残りから乱数種42で1,500件を選びます。公開確率 `p` を二値化せず教師とし、
`2 × |p - 0.5|` をセル単位の損失重みにします。したがって、0.5は損失へ寄与しません。

実行時には `label-source-validation.json` へ公開ラベルと正式58件の項目別AUCを保存し、
平均AUCが0.88未満の場合は画像学習を開始しません。`run-summary.json` には入力ハッシュ、
抽出検査のハッシュ、項目別教師統計、各世代の検証AUCを記録します。提出環境の
インターネットは無効のままです。

To change polling frequency, set `KAGGLE_POLL_SECONDS`, for example:

```bash
KAGGLE_POLL_SECONDS=60 ./scripts/wait.sh train
```

## Data boundary

- Input: `/kaggle/input[/competitions]/<competition-mount>` (read-only, Kaggle only)
- Training output: `/kaggle/working/outputs/best-model.pt` and its manifest
- Inference output: `/kaggle/working/outputs/submission.csv`
- Submission: `/kaggle/working/submission.csv` (required by the code competition)
- Local repository: code, configuration, and selected run outputs only

Do not add `kaggle competitions download`, `kagglehub.competition_download`, or a
copy step from `/kaggle/input` to the launcher. `.gitignore` also blocks common
medical-image formats and local `data/` directories as a final guardrail.

## Troubleshooting

- **Placeholder error:** finish step 4 above.
- **403 or access denied:** accept the competition rules in the browser and check
  `kaggle whoami`.
- **Input mount not found:** inspect the available mount names printed by the run,
  then update `competition.input_slug`.
- **GPU unavailable:** T4 is requested in metadata. GPU availability and quota are
  controlled by Kaggle; retry later or choose an accelerator available to the
  account.
- **Failed run:** view the run log on the Kaggle notebook page. Output retrieval is
  for files saved by a completed run and may be empty after an early failure.
