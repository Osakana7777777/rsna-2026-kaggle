# DINOv2 のランキング損失追加実験

2026年10月2日に Kaggle の版45で学習を起動し、実行中であることを確認した。
設定は `configs/dinov2-ranking.toml`。

起動コマンド：

```bash
./scripts/push.sh train configs/dinov2-ranking.toml
```

実行先：https://www.kaggle.com/code/osakanateishoku/rsna-2026-training
2026年10月3日に推論版31を正式提出した。提出識別子は56787504、公開スコアは0.899で完了。
提出記録は `submissions/0019-v31-dinov2-pairwise-ranking/README.md`。

- DINOv2 small、4 分割すべてを学習する。
- GPU 2 枚のデータ並列。各 GPU 4 症例、勾配蓄積 1 回、実効バッチサイズ 8。
- 線形重みの既存実験を基に、正解ラベルの重み 3.0、弱ラベルの重み
  `2 × |p − 0.5|` を使用する。弱ラベルは `llm_labels_v4_blend.csv`。
- 既存の重み付き二値交差エントロピーに、ランキング損失の 0.05 倍を足す。
  既存の陽性重みの自動計算は継続する。
- ランキング損失は同じ対象ラベルの `p > 0.5` と `p < 0.5` の症例ペアで
  `softplus(−(陽性の出力値 − 陰性の出力値))` を計算する。
  ペアの重みは両症例のラベル重みの積。対象ごとに重み付き平均を取り、
  有効なペアのある対象間で平均する。欠損、重み 0、`p = 0.5` は除外する。
- GPU 間の予測を勾配付きで共有し、全 8 症例のペアを使用する。
  有効なペアがないバッチではランキング損失は 0。
- 学習率、20 エポック、画像処理、モデル構成は線形重み実験を引き継ぐ。
  元の設定ファイルは変更しない。

起動せずにアップロード用の内容を生成するコマンド：

```bash
./scripts/prepare.sh train configs/dinov2-ranking.toml
./scripts/prepare.sh submission configs/dinov2-ranking.toml
```

将来この実験を起動するときは、設定ファイルを必ず明示する。
既定の `train` は従来の設定のまま。
