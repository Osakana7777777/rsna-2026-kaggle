# RSNA 2026：Codex → Kaggle・Runpod GPU 実験環境

Codex はこのリポジトリをローカルで編集します。学習とコードコンペティション用の
推論は、別々の非公開 Kaggle ノートブックとして実行します。学習ノートブックは
検証済みチェックポイントを公開し、提出ノートブックはその出力をマウントして
推論だけを行います。

コンペティションデータは、このコンピューターにはダウンロードしません。Kaggle は
`kaggle/kernel-metadata.json` の `competition_sources` に従い、実行時にデータを
`/kaggle/input` 以下へ読み取り専用入力として接続します。

## 構成

```text
rsna-2026-kaggle/
├── configs/baseline.toml          # 実験設定とマウント設定
├── kaggle/
│   ├── train-kernel-metadata.json # 学習ノートブックと入力
│   ├── kernel-metadata.json       # 推論・提出ノートブックと入力
│   └── launcher.py                # 遠隔実行の入口
├── scripts/
│   ├── prepare.sh                 # 学習・提出・EDA 用データを構築
│   ├── push.sh                    # 学習・提出ノートブックをアップロード
│   ├── status.sh                  # 最新実行状態を表示
│   ├── wait.sh                    # 完了または失敗まで監視
│   ├── output.sh                  # /kaggle/working の成果物を取得
│   ├── runpod.sh                  # Runpod Pod の作成と管理
│   ├── runpod-train.sh            # Runpod 内で学習を開始
│   └── verify.sh                  # 構成と安全方針を検査
├── submissions/                   # 各提出の詳細な記録
├── src/rsna2026/                  # 実験用パッケージ
├── outputs/                       # 取得した成果物、Git 管理対象外
└── .kaggle-build/                 # 生成したアップロード用データ、Git 管理対象外
```

## 初回設定

1. `uv` を導入し、公式 Kaggle コマンドラインクライアントをインストールします。

   ```bash
   uv tool install kaggle
   ```

2. Kaggle のアカウント設定画面で API トークンを作成し、Kaggle クライアントを
   設定します。トークンは必ずリポジトリの外で管理してください。次のコマンドで
   認証を確認します。

   ```bash
   kaggle whoami
   ```

3. Kaggle 上でコンペティション規約に同意します。API 認証だけでは
   コンペティションデータへアクセスできません。

4. このリポジトリを別のアカウントで使用する場合は、次の二つのメタデータファイルと
   `configs/baseline.toml` にアカウント名とコンペティション識別子を設定します。

   - 学習ノートブック：`kaggle/train-kernel-metadata.json`
   - 提出ノートブック：`kaggle/kernel-metadata.json`
   - コンペティションのマウント先：`configs/baseline.toml`

   コンペティション識別子は、マウントされるフォルダー名と一致させます。
   `/kaggle/input/<slug>` と `/kaggle/input/competitions/<slug>` の両方を
   自動検出します。Kaggle 側で異なるフォルダー名が使われる場合は、
   `competition_sources` には公式識別子を残したまま、`competition.input_slug` を
   実際のフォルダー名へ変更してください。

## 学習の実行

リポジトリのルートで次を実行します。

```bash
./scripts/verify.sh
./scripts/push.sh train
./scripts/status.sh train
./scripts/wait.sh train
./scripts/output.sh train
```

`osakanateishoku/rsna-2026-training` がアップロードされます。公開ソフトラベルと
コンペティションの学習画像を読み込み、モデルの学習と検証を行い、
`checkpoint-manifest.json` とともに `best-model.pt` を保存します。マニフェストには
チェックポイントの SHA-256 と正確なモデル構造が記録されます。

学習の来歴は `run-summary.json`、`label-source-validation.json`、
`slice-selection.json` に保存されます。学習ノートブックはテスト用表を読み込まず、
提出ファイルも作成しません。

## EDA の実行と成果物の取得

EDA ノートブックはコンペティションのマウントだけを使用し、GPU を必要としません。
表、図、代表的な DICOM 画像群を `/kaggle/working/outputs/eda` 以下へ保存します。
遠隔処理全体を実行するには次のコマンドを使います。

```bash
./scripts/run-eda.sh
```

非公開ノートブック `osakanateishoku/rsna-2026-eda` を構築してアップロードし、
完了を待ってから成果物をこのコンピューターの `outputs/eda/` へ取得します。
各段階を個別に実行する場合は、`./scripts/push.sh eda`、
`./scripts/wait.sh eda`、`./scripts/output.sh eda` を使用してください。

## 推論ノートブックの構築

学習ノートブックが正常終了した後にだけ、次を実行します。

```bash
./scripts/push.sh submission
./scripts/status.sh submission
./scripts/wait.sh submission
./scripts/output.sh submission
```

`osakanateishoku/rsna-2026-workflow` がアップロードされます。このノートブックは
`kernel_sources` を通じて `osakanateishoku/rsna-2026-training` の保存済み出力を
マウントします。推論前に、チェックポイントのファイル名、SHA-256、目的変数の順序、
画像寸法、モデル構造を検証します。その後、テスト用表だけを読み込み、
`/kaggle/working/submission.csv` を作成します。

新しい学習版を作成した後は、必ず提出ノートブックも新しい版としてアップロードして、
ノートブック入力を新しい学習出力へ固定してください。

学習と推論は同じソースコードと設定を内包しますが、生成時の実行方式はそれぞれ
`train` と `inference` に固定されます。設定を編集しても、提出ノートブックが
学習処理へ入ることはありません。

## モデルと前処理

モデルは各シリーズを、撮像断面と撮像法からなる六つの区分へ分類します。各シリーズを
DICOM の患者座標軸上で並べ、隣接する三断面の窓を作成します。利用可能な撮像範囲に
等間隔で配置した物理中心に最も近い、一意な窓を最大十個選択します。

DINOv2 はすべての有効な窓を一回のモデル呼び出しで符号化します。目的変数ごとの
マスク付き注意機構により、最初に各区分内の窓を集約し、その後に六区分を集約します。
各窓には物理位置と隣接間隔の特徴も付加します。重複する窓に含まれる画像も、復号・
寸法変更後の元画像を一度だけキャッシュします。学習時のデータ拡張は同一区分の全窓で
同期し、検証時と推論時は決定的な処理を行います。すべての実験設定は
`configs/baseline.toml` に定義されています。

Kaggle の実行環境から PyTorch、pandas、NumPy、pydicom が提供されます。実行中の
インターネット接続は無効のままで、基準実験はネットワーク経由の重み取得を必要としません。

## Kaggle への提出

推論が正常終了したら、その版番号を指定して提出します。

```bash
kaggle competitions submit rsna-knee-abnormality-detection \
  -k osakanateishoku/rsna-2026-workflow \
  -v VERSION \
  -f submission.csv \
  -m "Multi-view CNN baseline"
```

正式提出ごとに、`submissions/` 以下へ詳細記録を作成します。
`submissions/TEMPLATE.md` を複製し、提出順、Kaggle 版、短い実験名を含む
ディレクトリ名を付けてください。正確な設定、処理、検証、得点、既知の制限、
次の実験を記録します。得点が確定したら `submissions/README.md` も更新します。

学習から推論までの一連の処理は次のとおりです。

```bash
./scripts/push.sh train && ./scripts/wait.sh train && ./scripts/output.sh train
./scripts/push.sh submission && ./scripts/wait.sh submission && ./scripts/output.sh submission
```

## Runpod での学習

Runpod は Kaggle に代わる学習環境として使用し、提出は引き続き Kaggle で行います。

### runpodctl の導入

macOS では Homebrew を使う方法が簡単です。

```bash
brew install runpod/runpodctl/runpodctl
runpodctl version
```

macOS または Linux で公式導入スクリプトを使う場合は、次を実行します。

```bash
bash <(curl -sL cli.runpod.io)
runpodctl version
```

管理者権限がない場合は `~/.local/bin` に導入されることがあります。その場合は、
表示された案内に従って実行パスへ追加してください。導入後の更新には
`runpodctl update` を使用できます。

### API キーと SSH の設定

Runpod の設定画面にある API Keys からキーを作成します。キーには Pod の作成・確認・
停止・削除に必要な権限だけを付与し、パスワードと同様にリポジトリ外で管理してください。

初回設定には、API キーと SSH 鍵を対話形式で設定・検査する次のコマンドを推奨します。

```bash
runpodctl doctor
```

API キーだけを手動設定する場合は次を使用します。シェル履歴へキーが残る可能性があるため、
通常は `runpodctl doctor` を使用してください。

```bash
runpodctl config --apiKey YOUR_RUNPOD_API_KEY
```

既存の公開 SSH 鍵を追加して確認する場合は次のとおりです。

```bash
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
runpodctl ssh list-keys
```

SSH 鍵がない場合は、先に作成します。

```bash
ssh-keygen -t ed25519 -C "YOUR_EMAIL@DOMAIN.COM"
```

設定後、次のコマンドが正常に結果を返すことを確認します。

```bash
runpodctl version
runpodctl user
runpodctl gpu list
```

### GPU の選択と Pod の作成

在庫のある GPU の識別子を確認し、GPU の種類と枚数を明示して Pod を作成します。

```bash
./scripts/runpod.sh gpus
./scripts/runpod.sh create \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 2 \
  --network-volume-id YOUR_NETWORK_VOLUME_ID
```

`./scripts/runpod.sh gpus` は `runpodctl gpu list` を呼び出します。表示結果の
`gpuId` を `--gpu-id` へ正確に指定してください。現在の Pod 一覧と作成した Pod の
詳細は次のコマンドでも確認できます。

```bash
runpodctl pod list
runpodctl pod get POD_ID
```

`--dry-run` を指定すると、Pod を借りずに完全な作成コマンドを確認できます。
既定値は Secure Cloud、GPU 二枚、Runpod の PyTorch 画像、コンテナ領域 50 GB、
永続ボリュームのマウント先 `/workspace` です。ネットワークボリューム識別子を
指定しない場合は、破棄可能な 200 GB の Pod ボリュームを作成します。

Pod は停止または削除するまで課金されます。状態の確認、停止、削除には次を使います。

```bash
./scripts/runpod.sh get POD_ID
./scripts/runpod.sh stop POD_ID
./scripts/runpod.sh delete POD_ID
```

Pod の起動後、SSH 接続コマンドは次のように取得できます。

```bash
runpodctl ssh info POD_ID
```

表示された `ssh` コマンドをそのまま実行して接続します。Pod 作成後に SSH 公開鍵を
Runpod へ追加した場合、その鍵は起動済み Pod に自動反映されないことがあります。
その場合は Pod を再作成するか、Runpod の画面上の端末から公開鍵を追加してください。

スクリプトはコンペティションデータを複製しません。コンペティション規約で外部計算環境への
配置が許可されたデータだけを用意してください。永続ボリューム上の既定入力構成は
次のとおりです。

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
├── rsna-2026-preprocess-shard-0/outputs/image-cache/
│   ├── cache-manifest.json
│   └── studies/*.npz
├── rsna-2026-preprocess-shard-1/outputs/image-cache/
│   ├── cache-manifest.json
│   └── studies/*.npz
├── rsna-2026-preprocess-shard-2/outputs/image-cache/
│   ├── cache-manifest.json
│   └── studies/*.npz
├── rsna-2026-preprocess-shard-3/outputs/image-cache/
│   ├── cache-manifest.json
│   └── studies/*.npz
└── dinov2-small/
    ├── config.json
    └── model.safetensors（または同等のローカルモデル重みファイル）
```

Pod へ接続した後、このリポジトリを `/workspace` 以下へ配置し、`uv` を導入して、
リポジトリ内から学習を開始します。実行スクリプトは PyTorch 画像のシステムパッケージを
参照できる `uv` 仮想環境を作成します。必要に応じて `pydicom` と `transformers` を
導入し、CUDA を検証して、既定では認識されたすべての GPU を使用します。

```bash
./scripts/runpod-train.sh \
  --input-root /workspace/input \
  --output-dir /workspace/outputs/experiment-name
```

Pod に割り当てられた GPU の一部だけを使用する場合は、`--gpu-count 1` など、
認識された枚数以下の値を指定します。固定済みの断面選択キャッシュを配置していない場合は、
`--build-selection-cache` を指定してください。学習前に配置済み DICOM のヘッダーから
キャッシュを再構築します。

四つの `rsna-2026-preprocess-shard-*` には、Kaggleで生成した前処理済み画像キャッシュを
ディレクトリ構成ごと配置します。学習開始時に `configs/baseline.toml` に記録された
マニフェストのSHA-256と全検査の圧縮配列を検証します。この構成では学習中に元DICOMを
復号せず、Kaggle学習時と同じ8ビット前処理画像を読み込みます。四分割の合計容量は約
24.76 GiBです。

実行環境固有のパスと GPU 枚数は実行時に上書きされるため、
`configs/baseline.toml` は Kaggle 実験と同一の定義を維持します。

## 実験0002：公開 LLM 混合ラベル

学習ノートブックは公開データセット
`stevenleehans/rsna-knee-llm-report-labels` の
`llm_labels_v4_blend.csv` を入力として参照します。ファイルの SHA-256、4,407行、
十二所見の列、識別子集合、欠損、値域を学習前に検証し、公開データセットの更新へ
自動追従しません。

正式ラベル58件はすべて検証専用です。同じ正規化報告文を持つ検査を候補から外し、
残りから乱数種42で1,500件を選びます。公開確率 `p` を二値化せず教師とし、
`2 × |p - 0.5|` をセル単位の損失重みにします。したがって、0.5は損失へ寄与しません。

実行時には `label-source-validation.json` へ公開ラベルと正式58件の項目別 AUC を保存し、
平均 AUC が0.88未満の場合は画像学習を開始しません。`run-summary.json` には入力ハッシュ、
抽出検査のハッシュ、項目別教師統計、各世代の検証 AUC を記録します。提出環境の
インターネット接続は無効のままです。

監視間隔を変更するには、次のように `KAGGLE_POLL_SECONDS` を設定します。

```bash
KAGGLE_POLL_SECONDS=60 ./scripts/wait.sh train
```

## データ境界

- 入力：`/kaggle/input[/competitions]/<competition-mount>`。Kaggle 上でのみ読み取り専用
- 学習出力：`/kaggle/working/outputs/best-model.pt` とそのマニフェスト
- 推論出力：`/kaggle/working/outputs/submission.csv`
- 提出ファイル：`/kaggle/working/submission.csv`。コードコンペティションの必須位置
- ローカルリポジトリ：コード、設定、選択して取得した実行成果物だけ

`kaggle competitions download`、`kagglehub.competition_download`、
`dataset_download`、または `/kaggle/input` からランチャーへの複製処理を追加しないで
ください。`.gitignore` でも一般的な医用画像形式とローカルの `data/` ディレクトリを
除外し、最後の安全策としています。

## 問題への対処

- 設定用の仮文字列に関するエラー：初回設定の手順4を完了してください。
- 403またはアクセス拒否：ブラウザーでコンペティション規約へ同意し、
  `kaggle whoami` を確認してください。
- 入力マウントが見つからない：実行ログに表示されたマウント名を確認し、
  `competition.input_slug` を更新してください。
- GPU を利用できない：メタデータでは T4 を要求しています。GPU の空きと割り当て上限は
  Kaggle が管理するため、時間を置いて再実行するか、利用可能な演算装置を選択してください。
- 実行失敗：Kaggle のノートブック画面で実行ログを確認してください。成果物取得は
  完了した実行が保存したファイルを対象とするため、早期失敗時には空になる場合があります。
