# Repository Guidelines

This repository contains code for Kaggle's [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition.

## Project Structure & Module Organization

Core code lives in `src/rsna2026/`: `config.py` loads settings, `data.py` validates Kaggle mounts, and `train.py` is the experiment entry point. Put settings in `configs/`; `baseline.toml` is the default. `kaggle/launcher.py` starts the packaged job, while `kernel-metadata.json` controls its environment. Shell workflows belong in `scripts/`. Generated `.kaggle-build/`, `outputs/`, `.venv/`, and competition data must remain untracked.

## Build, Test, and Development Commands

Use `uv` for every Python environment; do not create environments with `venv`, Conda, or Poetry.

```bash
uv venv .venv                 # create the local Python environment
./scripts/verify.sh            # validate files, shell/Python syntax, config, and data policy
./scripts/prepare.sh           # assemble the Kaggle upload payload
./scripts/push.sh              # prepare, upload, and launch a private Kaggle run
./scripts/status.sh            # inspect the latest remote run
./scripts/wait.sh              # poll until the run finishes
./scripts/output.sh            # retrieve /kaggle/working artifacts
```

Configure all `YOUR_...` placeholders before pushing. Run remote jobs only after `kaggle whoami` succeeds and competition rules are accepted.

## Coding Style & Naming Conventions

Target Python 3.10 or newer. Follow PEP 8 with four-space indentation, type annotations, concise docstrings, `snake_case` functions and modules, and `UPPER_CASE` constants. Keep imports explicit. Shell scripts must use Bash, begin with `set -euo pipefail`, quote expansions, and reuse `scripts/common.sh`. Use lowercase, descriptive TOML keys.

## Testing Guidelines

No dedicated test suite or coverage threshold exists yet. Add tests under `tests/` as `test_<module>.py`; prefer `pytest` if a suite is introduced, and add it through `uv`. Cover path validation, configuration parsing, and output generation without downloading or reading medical images. Before every change submission, run `./scripts/verify.sh`. For workflow changes, also inspect the payload produced by `./scripts/prepare.sh`.

## Commit & Pull Request Guidelines

The repository has no commit history from which to infer a convention. Use short, imperative subjects such as `Add mount validation` and keep each commit focused. Pull requests should explain the experiment or workflow change, list validation commands run, identify configuration changes, and link relevant issues. Include logs or output summaries for Kaggle behavior changes; include screenshots only for visual artifacts. Never commit credentials, API tokens, competition data, checkpoints, or generated build contents.

## Data & Security Boundaries

Read competition input only from `/kaggle/input/<mount>` and write retrievable artifacts only beneath `/kaggle/working/outputs`, except for the code-competition-required `/kaggle/working/submission.csv`. Keep Kaggle credentials outside the repository. Do not add local competition-download commands or copy input data into the upload payload.

## Submission Documentation

For every formal Kaggle submission, create
`submissions/<sequence>-v<kernel-version>-<experiment>/README.md` from
`submissions/TEMPLATE.md`. Write the record in Japanese and include the exact code and
configuration behavior, data selection, preprocessing, model, training, validation,
inference, submission command, Kaggle version, submission identifier, public score,
known limitations, and next experiments. Update `submissions/README.md` as the index.
