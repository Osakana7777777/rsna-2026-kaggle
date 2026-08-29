調べた範囲では、膝MRIのマルチ断面・マルチseries分類では、**「巨大な3D CNNをそのまま入れる」より、2D/2.5D backboneで各slice・局所volumeを表現し、slice/series/plane方向をattentionやTransformerで集約する構成**が現在かなり有力です。特に今回のRSNA Knee 2026を念頭に置くなら、モデル構造そのもの以上に、**DICOM幾何に基づく前処理・病変ごとのsequence選択・report由来weak labelの質**が性能を左右する可能性が高いです。

## まず結論

今回のタスクなら、私なら最終的に次の構造を狙います。

```text
Study
│
├─ Sagittal fluid-sensitive series
│     └─ 2.5D slices → DINOv2-S/B
│
├─ Coronal fluid-sensitive series
│     └─ 2.5D slices → DINOv2-S/B
│
├─ Axial fluid-sensitive series
│     └─ 2.5D slices → DINOv2-S/B
│
└─ T1 / non-fat-sat series
      └─ 2.5D slices → DINOv2-S/B

              ↓
      slice positional encoding
              ↓
       slice attention / Transformer
              ↓
         series embedding
              ↓
   pathology-specific series attention
              ↓
        12個のclassification head
```

つまり

**2.5D DINOv2 → slice aggregation → sequence/plane fusion → pathology-specific attention**

です。

フル3Dモデルもensemble候補として試す価値はありますが、**主力モデルにする優先度は2.5D+Transformerより低い**と考えます。

---

# 1. なぜ単純2Dでは不十分なのか

古典的なMRNetは非常に参考になります。MRNetでは各MRI sliceをAlexNetに独立に入れ、

\[
x_1,\ldots,x_N
\rightarrow
f_1,\ldots,f_N
\]

と特徴抽出し、その後

\[
f_{\mathrm{series}}
=\max_i f_i
\]

という単純なslice-wise max poolingを行っています。その後Sagittal / Coronal / Axialの予測をlogistic regressionで統合しました。

それでもMRNetは、

- abnormality AUC 0.937
- ACL tear AUC 0.965
- meniscal tear AUC 0.847

を達成しており、**「2D backbone + study-level pooling」だけでもかなり強い**ことを示しました。

ただしmax poolingには明確な問題があります。

```text
slice 1 ─┐
slice 2 ─┤
slice 3 ─┤
 ...     ├─ MAX → Study prediction
slice N ─┘
```

これでは、

- sliceの順番
- 隣接sliceとの連続性
- 病変が複数sliceにまたがること
- 「tearがsurfaceまで連続する」といった3D構造

をほぼ捨てています。

今回ならMRNetは**最低限のbaseline**としては優秀ですが、最終形ではありません。

---

# 2. 2.5Dがかなり相性がよい

膝MRIは典型的な**anisotropic volume**です。

つまり、

\[
\Delta x,\Delta y \ll \Delta z
\]

になりやすく、

```text
in-plane
0.3–0.6 mm

slice direction
3–4 mm
```

のようなデータが珍しくありません。

このような画像を通常の3D convolutionで

\[
3\times3\times3
\]

として扱うと、x,y,z方向を同じ距離のように扱ってしまいます。

医用画像分野でも、anisotropic MRIでは

- pure 2D → slice間情報を失う
- pure 3D → anisotropy、計算量、小規模データで不利
- 2.5D → 両者の妥協点

という評価があります。

そこで例えば、

```text
slice i-1
slice i
slice i+1
```

をRGB channelのようにして、

\[
X_i =
[I_{i-1},I_i,I_{i+1}]
\in \mathbb R^{3\times H\times W}
\]

としてDINOv2やConvNeXtに入力します。

今回の公開baselineでも、

**3 adjacent slices / input slot**

という2.5D方式が使われています。

これはかなり理にかなっています。

---

# 3. BackboneはDINOv2が第一候補

これは今回かなり明確です。

自然画像ImageNet CNNだけでなく、

- RadImageNet
- DINOv2
- domain-specific self-supervised learning

が候補になります。

RadImageNetは135万枚のCT/MRI/USから作られた医用画像事前学習データで、ACL・meniscus MRIを含む複数のdownstream taskでImageNet初期化との比較が行われています。

一方、今回のRSNA Kneeでは**DINOv2が既に非常に強い**です。8月29日時点のKaggle上ではDINOv2を使う公開Notebook群にpublic score約0.92台のものが確認できます。

さらに最新の公開Notebook「Head and shoulders, knees and toes」はBest Score 0.936を表示しています。ただしpublic leaderboardはvalidationではなく、モデル詳細やensemble・label生成方法も絡むので、この数字だけからarchitectureの優劣を結論づけるべきではありません。

### なぜDINOv2がMRIでも効き得るか

DINOv2はclassification labelではなくself-supervised learningによって、

\[
\text{image}
\rightarrow
\text{generic local/semantic representation}
\]

を学習しています。

膝MRIの場合も、

- 骨輪郭
- cartilage
- meniscus
- ligament
- marrow
- fluid
- texture boundary

などの構造をpatch representationとして扱えるため、少数ラベル環境と相性がよいです。

今回のように**gold labelが58 studyしかない**状況では特に重要です。

---

# 4. ただし「DINOv2にするだけ」では勝てない

ここは今回のcompetitionから得られる非常に重要な知見です。

Kaggle discussionでは、DINO系single modelが0.91–0.92程度に到達する一方、

> encoderを大きくしてもCVが改善しない  
> crop geometry、resolution、slice selectionの改善が効く

という報告が出ています。上位参加者からもlabel qualityを重要視するコメントがあります。

つまり、

\[
\boxed{
\text{better input}
>
\text{larger backbone}
}
\]

となる可能性があります。

これはMRIでは特に納得できます。

---

# 5. DICOM geometry処理がかなり重要

今回ここを私はかなり重視します。

sliceのファイル名やInstanceNumberを盲目的に信用するより、

\[
\mathbf n =
\mathbf r \times \mathbf c
\]

としてImageOrientationPatientからslice normalを求め、

\[
z_i =
\mathbf p_i\cdot\mathbf n
\]

でImagePositionPatientを投影して並べる方が物理的に正しいです。

公開baselineの分析でも、

- filenameはslice順序を保証しない
- DICOM orientation / positionでorderすべき
- physical cropを使う

ことが推奨されています。

これは以前扱った

`ImagePositionPatient`
`ImageOrientationPatient`

がまさに生きてくる部分です。

---

# 6. pixel cropではなくmm crop

かなり重要なポイントです。

例えば全studyで

```python
CenterCrop(336)
```

とすると、PixelSpacingが違うため、

```text
Patient A
336 pixel = 100 mm

Patient B
336 pixel = 170 mm
```

のようになり得ます。

つまりモデルが見る**実際の解剖学的FOVが異なる**。

そこで、

\[
\text{crop width pixels}
=
\frac{\text{desired FOV mm}}
{\text{PixelSpacing}}
\]

として、

```text
130 mm × 130 mm
```

など固定physical-FOVをcropしてからresizeする方が合理的です。

公開baselineでは

**130 mm physical crop → 336 px**

が使われています。

特にmeniscus tearやACLのような小病変ではかなり効く可能性があります。

---

# 7. slice poolingはMaxよりAttention/Transformer

ここは次に改善すべき場所です。

### MRNet

\[
z=\max_i f_i
\]

### 改良版

各sliceに位置情報を加えて、

\[
h_i=f_i+p_i
\]

とし、

\[
h_1,\ldots,h_N
\overset{\mathrm{Transformer}}{\longrightarrow}
h_{\mathrm{CLS}}
\]

とします。

実際、2025年European Radiologyの膝MRI研究では、

```text
MRI sequence
↓
256×256 × 32 slices
↓
modified ResNet18
↓
局所3D residual convolution
↓
slice token
↓
positional embedding
↓
Transformer encoder
↓
23 condition classification
```

という構成が使われています。3,121 studyで学習し、448 studyの外部testも行っています。

これは今回の問題にかなり直接的な先行例です。

---

# 8. 「局所3D + 大域Transformer」が特に面白い

この2025年モデルの設計はよく考えられています。

完全な3D CNNではなく、

```text
slice i-1
slice i
slice i+1
   ↓
local 3D convolution
   ↓
slice token
```

として、近傍sliceだけ3D convolutionで処理します。

その後

```text
token 1
token 2
...
token 32
    ↓
Transformer
```

として遠距離slice関係を扱います。

したがって今回なら、

\[
\boxed{
\text{2D pretrained encoder}
+
\text{local 3D context}
+
\text{global attention}
}
\]

はかなり魅力的です。

私なら純粋2.5Dの次にこれを試します。

---

# 9. Sagittal / Coronal / Axialを単純平均しない

ここも重要です。

臨床的にもplaneごとに情報量が違います。

ESSRは膝MRIをSagittal / Coronal / Axialの3方向で評価することを推奨しています。さらにT1とfluid-sensitive sequenceの双方が必要とされています。

例えば概念的には、

| 病変 | 特に重要になりやすい情報 |
|---|---|
| ACL | sagittal、fluid-sensitive |
| MCL | coronal |
| Medial/Lateral meniscus | sagittal + coronal、高分解能PD |
| Medial/Lateral OA | coronal + sagittal |
| PF OA | axial + sagittal |
| Effusion | sagittal + axial、fluid-sensitive |
| Baker cyst | axial/sagittal |
| Contusion | fluid-sensitive、多方向 |
| Fracture | T1 + fluid-sensitive、多方向 |

といった具合です。

SagittalではACLを全長に近く描出できるよう撮像方向を設計することが推奨され、PD-weighted sagittal/coronalはmeniscus評価で重要です。Patellofemoral cartilageは特にaxialの情報価値が高いです。

したがって、

\[
z=
\frac{
z_{\mathrm{sag}}
+z_{\mathrm{cor}}
+z_{\mathrm{ax}}
}{3}
\]

より、

\[
z_k
=
\sum_s
\alpha_{k,s}z_s
\]

とする方が自然です。

ここで

- \(k\)：病変
- \(s\)：sequence/plane
- \(\alpha_{k,s}\)：病変ごとのattention weight

です。

---

# 10. Pathology-specific attention

これは今回かなり有望だと思います。

例えばseries embeddingを

\[
e_{\text{sag,FS}},
e_{\text{cor,FS}},
e_{\text{ax,FS}},
e_{\text{sag,T1}},...
\]

と作ります。

そしてACL headなら

\[
\alpha^{ACL}
=
\operatorname{softmax}
(W_{ACL}E)
\]

として、

\[
z_{ACL}
=
\sum_s
\alpha^{ACL}_s e_s
\]

とする。

Meniscusなら別のattention、

\[
\alpha^{Meniscus}
\]

を使います。

つまりモデル自身に

> 「ACLを見るときはSagittalを重視しろ」  
> 「PF OAを見るときはAxialを重視しろ」

を学ばせるわけです。

今回の公開baselineでも**per-diagnosis attention over slots**が設計上重要な要素として使われています。

これは12病変を単純なshared global embeddingから12 sigmoidに出すより合理的です。

---

# 11. Multi-plane fusionには実証的な裏付けもある

TripleMRNetという最近の研究では、

- sagittalのみ
- coronalのみ
- axialのみ
- sagittal + coronal
- sagittal + axial
- coronal + axial
- 3 planes

の7構成を比較しています。

ACLでは3-plane modelがaccuracy 0.925、sensitivity 0.944など最も良い結果を出しました。

したがって、

> 「ACLならSagittalだけで十分」

と最初から決め打ちするより、

**病変ごとに主planeはあるが、他planeを捨てない**

方がよさそうです。

---

# 12. Full 3D CNNはどうか

3D ResNet / 3D EfficientNet / 3D ConvNeXtなども当然候補です。

メリットは、

\[
I(x,y,z)
\]

を直接処理できること。

ただし今回のMRIは、

- slice厚が異なる
- seriesごとにslice数が違う
- planeが違う
- sequence contrastが違う
- scanner/site差が大きい

ので、

```text
全部を1個の3D tensor
```

にするのは実はかなり無理があります。

また5000 study程度、しかもclean labelはごく少数という条件では、3Dモデルはparameter/data efficiencyでも不利です。

2024年のsemi-supervised knee OA研究では3D CNN + multiviewの有効性は示されていますが、今回のようなheterogeneous routine MRIでは必ずしも最善とは限りません。

したがって私は

**3D CNN = diversityを出すensemble member**

として使います。

---

# 13. Multi-task learningも有望

12病変は独立ではありません。

例えば、

\[
ACL\ tear
\rightarrow
bone\ contusion
\]

や、

\[
OA
\leftrightarrow
effusion/synovitis
\]

など相関があります。

したがって

```text
shared encoder
↓
shared study representation
↓
12 pathology-specific heads
```

は自然です。

さらに一歩進めて、

```text
classification
+
segmentation/localization
```

を同時学習させる方法もあります。

2025年のmeniscus MRI研究では、meniscus segmentationを補助taskにするmulti-task networkがsingle-task modelを上回り、Sagittal + Coronal fusionもsingle-planeより良好でした。

今回segmentation GTはありませんが、

- external segmentation model
- pseudo-mask
- anatomy localization

を補助signalとして使う方向は研究価値があります。

---

# 14. 実は今回最大の問題はlabel

ここはarchitecture以上に重要です。

今回のtrainには4,407 studyがありますが、公式12ラベルがあるのは58 studyのみで、残り4,349 studyにはreportしかありません。

さらにreportは多言語です。公開調査ではEnglishだけでなくTurkish、Spanish、German、Greek、Dutch、Frenchなどが確認されています。

しかもhidden testでは**reportは提供されない**ことをcompetition hostが明言しています。

したがって、

```text
TRAIN

report ─────────────┐
                    ↓
MRI ─────────→ image model
                    ↓
                prediction


TEST

MRI ─────────→ image model
```

という構造にする必要があります。

reportは**inference featureではなくteacher**として使うべきです。

---

# 15. Soft weak labelがよい

Reportから

```text
ACL = 0/1
```

とhard pseudo-labelを作るより、

```text
ACL = 0.92
MCL = 0.05
Effusion = 0.73
...
```

というsoft targetにする方がよいでしょう。

例えば

\[
L
=
w_i\,
\operatorname{BCE}(p_i,\tilde y_i)
\]

として、

gold:

\[
w_i=5\sim10
\]

weak:

\[
w_i=1
\]

のように重み付けできます。

公開baselineでも、

- report-derived weak labels
- 58 reviewed studiesを高weight

とする方式が採用されています。

Kaggle discussionでも上位参加者が**soft-label qualityが主要bottleneck**と指摘しています。

今回これはかなり重要でしょう。

---

# 16. Self-supervised domain adaptationもかなり面白い

さらに一段進めるなら、

4,407 studyの全MRI sliceを使ってDINO/MAEを自己教師ありで追加学習します。

つまり、

```text
ImageNet DINOv2
      ↓
RSNA Knee MRIでSSL
      ↓
12-label fine tuning
```

です。

これならreport label noiseに依存せず、

\[
p(x)
\]

そのものから膝MRI固有のrepresentationを学習できます。

最近の医用pretraining研究でも、modality/anatomy awareなcontrastive pretrainingがMRNet ACL downstreamでAUC 0.964を報告しています。

ただし今回のpublic communityを見る限り、

> DINOを大きくすれば自動的に上がる

わけではありません。

したがってdomain SSLは、geometry pipelineを固めてからです。

---

# 17. 私なら実験をこの順番にする

今回のコンペにそのまま落とすなら、以下の順番がかなり効率的だと思います。

**B0：2.5D CNN baseline**

```text
Sag / Cor / Ax
3 adjacent slices
↓
EfficientNet-B0
↓
mean/max pooling
↓
12 labels
```

まずpipeline確認。

**B1：2.5D DINOv2-S**

```text
3 adjacent slices
↓
DINOv2-S
↓
mean pooling
```

ここでCNN vs DINOを比較。

**B2：geometry-aware DINOv2**

```text
ImageOrientationPatient
ImagePositionPatient
PixelSpacing
↓
physical slice ordering
physical crop
laterality normalization
↓
DINOv2
```

私はここがB1→B2でかなり重要だと考えています。

**B3：Attention MIL**

```text
slice embeddings
↓
attention pooling
↓
series embedding
```

MRNetのmax poolingを置き換えます。

**B4：Slice Transformer**

```text
e1 + pos1
e2 + pos2
...
eN + posN
↓
Transformer
```

**B5：Multi-series pathology-specific attention**

```text
Sag-FS ─┐
Cor-FS ─┤
Ax-FS  ─┤→ pathology-specific attention → 12 heads
Sag-T1 ─┘
```

これはかなり本命。

**B6：local 3D residual + Transformer**

2025 European Radiology型です。

```text
ResNet/DINO
+
local 3D conv
+
slice Transformer
```

**B7：ensemble**

最後に

```text
DINOv2
+
RadImageNet CNN
+
3D/Res3D model
```

をrank averaging。

今回metricが12 label macro ROC-AUCであるため、rank averagingとの相性もよいです。公開分析でもAUCがmonotonic transformationに不変であることからrank meanが採用されています。

---

# 18. 今回特に優先度が低そうなもの

逆に、今すぐ私はあまりcomputeを投入しません。

**巨大ViT**はDINOv2-S → B/Lと大きくしても改善が保証されず、communityでもbackbone capacityよりinput/label qualityが問題という報告があります。

**巨大3D CNN**もanisotropy、heterogeneous sequence、小規模clean supervision、VRAM/runtimeの点で不利です。

**全slice uniform samplingだけ**も病変が局所的なので改善余地が大きいです。

**全series単純平均**も病変によって情報価値が違うので勿体ないです。

---

# 19. 現時点でのモデル優先順位

私なら優先度をこう置きます。

\[
\boxed{
\textbf{1位:
2.5D DINOv2
+
Slice Transformer
+
pathology-specific multi-series attention}
}
\]

次点が、

\[
\boxed{
\textbf{2位:
2D backbone
+
local residual 3D conv
+
Transformer}
}
\]

その次が、

\[
\boxed{
\textbf{3位:
RadImageNet CNN / ConvNeXt
+
Attention MIL}
}
\]

そしてdiversity用として、

\[
\boxed{
\textbf{4位:
3D ResNet / 3D ConvNeXt}
}
\]

です。

ただし**モデルより先に**

\[
\boxed{
\text{DICOM geometry}
+
\text{physical FOV}
+
\text{weak-label quality}
}
\]

を固めるべきです。

今回の公開competitionの状況も、まさにその方向を示しています。8月29日時点ではDINOv2系公開モデルが0.92前後〜それ以上まで到達していますが、参加者間ではモデルcapacityよりもcrop/slice selectionとreport-derived supervisionがbottleneckだという議論が強くなっています。

### 今回のRSNA Kneeに対する具体案

以前検討していたbaselineから自然に進めるなら、

```text
B0
EfficientNet-B0
3-slice 2.5D
各plane独立
mean/max pooling

      ↓

B1
DINOv2-S
3-slice 2.5D
physical 130mm crop
336×336

      ↓

B2
DINOv2-S
+ slice positional encoding
+ attention pooling

      ↓

B3  ← 本命
DINOv2-S/B
+ Slice Transformer
+ 6前後のsequence slots
+ 12 pathology-specific attention heads

      ↓

B4
B3
+ local 3D residual conv

      ↓

B5
B3/B4
+ RadImageNet CNN ensemble
+ fold/rank ensemble
```

という実験系がかなり筋が良いです。

特に次の一手としては、**B0を単に高性能backboneへ置換するのではなく、「physical-scale preprocessing → 2.5D DINOv2 → slice/series attention」までを一つの設計思想として作る**のがよいと思います。文献的にも、現在のcompetition実績的にも、この方向が最も根拠があります。
