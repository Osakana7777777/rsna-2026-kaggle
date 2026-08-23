# RSNA 2026: Codex → Kaggle GPU workflow

Codex edits this repository locally. The helper scripts assemble only code and
configuration into `.kaggle-build/`, upload it as a private Kaggle script, start
the GPU run, inspect its status, and retrieve the files written below
`/kaggle/working`.

Competition data is never downloaded to this computer. Kaggle attaches it to the
run as a read-only input below `/kaggle/input`, using `competition_sources` in
`kaggle/kernel-metadata.json`.

## Layout

```text
rsna-2026-kaggle/
├── configs/baseline.toml       # experiment and mount placeholders
├── kaggle/
│   ├── kernel-metadata.json    # Kaggle script, GPU, and competition input
│   └── launcher.py             # remote entry point
├── scripts/
│   ├── prepare.sh              # builds the upload-only payload
│   ├── push.sh                 # uploads and starts a Kaggle run
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

4. Replace both placeholders:

   - `YOUR_KAGGLE_USERNAME` in `kaggle/kernel-metadata.json`
   - `YOUR_RSNA_2026_COMPETITION_SLUG` in both
     `kaggle/kernel-metadata.json` and `configs/baseline.toml`

   The competition slug must also be the mounted competition folder name.
   Both `/kaggle/input/<slug>` and `/kaggle/input/competitions/<slug>` layouts
   are detected. If Kaggle uses a different folder name, set
   `competition.input_slug` to that name while leaving `competition_sources`
   as the official competition slug.

## Run and iterate

From this directory:

```bash
./scripts/verify.sh
./scripts/push.sh
./scripts/status.sh
./scripts/wait.sh
./scripts/output.sh
```

`push.sh` calls `prepare.sh` first. Because the Kaggle CLI uploads only the file
named by `code_file`, the preparation step embeds the current `src/` and config
into one generated notebook, as required for code-competition submissions.
Kaggle starts it automatically after a successful push.
`output.sh` downloads only notebook outputs, such as logs, summaries, checkpoints,
or `submission.csv`; it does not download the competition input.

The experiment reads public report-derived soft labels, selects sagittal, coronal,
and axial MRI series, and trains a compact multi-view convolutional network for
the twelve findings. The 58 fully annotated studies are validation-only. Label
confidence weights exclude undecided cells from the loss, and the best validation
epoch is used to predict the test studies. The run writes `submission.csv`,
`best-model.pt`, `label-source-validation.json`, `run-summary.json`, and `run.log`
below `/kaggle/working/outputs`.
For code-competition submission, `submission.csv` is also written at
`/kaggle/working/submission.csv`, the exact location required by Kaggle.

The model uses three slices per anatomical plane by default. These and the image
size, epoch count, batch size, learning rate, and public-label selection are set
in `configs/baseline.toml`.

The Kaggle image supplies PyTorch, pandas, NumPy, and pydicom. Internet access
remains disabled during the run, and the baseline does not require downloaded
weights.

After a successful code-competition run, submit the generated output by using
the version number printed by `push.sh`:

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

For one-command execution after configuration:

```bash
./scripts/push.sh && ./scripts/wait.sh && ./scripts/output.sh
```

## 実験0002: 公開LLM混合ラベル

提出ノートブックは公開データセット
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
KAGGLE_POLL_SECONDS=60 ./scripts/wait.sh
```

## Data boundary

- Input: `/kaggle/input[/competitions]/<competition-mount>` (read-only, Kaggle only)
- Output: `/kaggle/working/outputs` (retrievable with `output.sh`)
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
