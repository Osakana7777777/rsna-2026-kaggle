読み込むと、TransMILの本質は単に「MILのpoolingをTransformerに置き換えた」よりも、**MILを「各instanceの重要度を決める問題」から「instance同士の関係を学習する問題」に拡張した**点にあります。

論文本文と公開実装を突き合わせると、今回のRSNA Kneeにかなり参考になります。一方で、そのままMRIへ移植すべきではない部分もあります。

## 1. まずTransMILが問題視しているもの

通常のAttention-based MIL、たとえばIlse型なら、instance \(h_i\) ごとに

\[
a_i = \mathrm{attention}(h_i)
\]

を計算して、

\[
z=\sum_i a_i h_i
\]

とします。

つまり、

```text
patch 1 ─→ importance 0.1 ┐
patch 2 ─→ importance 0.7 ├→ weighted sum → classification
patch 3 ─→ importance 0.2 ┘
```

です。

TransMILの著者らが問題視するのは、

> patch 2 が重要かどうかを判断するとき、patch 1や3との関係そのものを十分モデル化していない

ことです。

そこで、

\[
h'_i=f(h_i,h_1,\ldots,h_n)
\]

として、**各instanceを他instanceとの関係を踏まえて更新する**。そのためにself-attentionを使います。論文ではこれを「correlated MIL」と呼んでいます。

これは膝MRIで言えば、

```text
ABMIL:
このsagittal window単独でACL tearらしいか？

TransMIL:
このwindowの所見は、
隣接windowや別部位のwindowと組み合わせると
ACL tearとして整合的か？
```

という違いです。

---

# 2. 論文が考える「pooling matrix」

ここがTransMILを理解するうえでかなり重要です。

著者らはMIL poolingを行列 \(P\) として解釈します。

従来型では概念的に

\[
P=
\begin{pmatrix}
a_1&0&0\\
0&a_2&0\\
0&0&a_3
\end{pmatrix}
\]

のような**対角行列**です。

つまり各instance自身の重みしかありません。

一方self-attentionでは、

\[
P=
\begin{pmatrix}
p_{11}&p_{12}&p_{13}\\
p_{21}&p_{22}&p_{23}\\
p_{31}&p_{32}&p_{33}
\end{pmatrix}
\]

となり、非対角成分

\[
p_{ij}
\]

が

> instance \(i\) が instance \(j\) をどれくらい参照するか

を表します。

論文Figure 2がまさにこの考え方です。

なのでTransMILにとって、

**attention pooling**

と

**self-attention**

はかなり違います。

---

# 3. 実際のTransMILの全体構造

論文のモデルは驚くほどシンプルです。

\[
\text{CNN features}
\rightarrow
\text{FC}
\rightarrow
\text{Transformer}
\rightarrow
\text{PPEG}
\rightarrow
\text{Transformer}
\rightarrow
[\mathrm{CLS}]
\rightarrow
\text{classifier}
\]

です。

公開実装では、

```text
ResNet50
 ↓
1024 dimensional patch feature

Linear(1024 → 512)
ReLU
 ↓
512 dimensional tokens

CLS token追加
 ↓
TransLayer 1
 ↓
PPEG
 ↓
TransLayer 2
 ↓
LayerNorm
 ↓
CLS token
 ↓
Linear(512 → classes)
```

となっています。

コードそのものも、

```python
h = self._fc1(h)

h = self.layer1(h)

h = self.pos_layer(h, _H, _W)

h = self.layer2(h)

h = self.norm(h)[:, 0]

logits = self._fc2(h)
```

です。

---

# 4. 最初にResNetのpatch featureを使う

WSIを256×256のpatchに切って、各patchをResNet50に入れ、

\[
x_i\in\mathbb R^{1024}
\]

というfeature vectorにします。

その後、

\[
1024\rightarrow512
\]

のlinear layerを通します。

したがってTransformerがpixelを直接見るわけではありません。

```text
WSI patch
  ↓
ResNet50
  ↓
embedding
  ↓
Transformer MIL
```

です。

これはまさに今のあなたの

```text
MRI 2.5D window
  ↓
DINOv2 / CoAtNet
  ↓
embedding
  ↓
MIL head
```

に対応します。

論文では平均8,800〜15,000程度ものpatchが1 slideに含まれるため、この後のattention計算が大問題になります。

---

# 5. CLS tokenを1個追加する

ViTと同じようにlearnableな

\[
h_{\mathrm{CLS}}
\]

を先頭に追加します。

```text
CLS  h1 h2 h3 ... hn
```

Transformerを2層通したあと、

```python
h = self.norm(h)[:, 0]
```

としてCLSだけを抜き、slide classificationをします。公開実装でもCLS tokenは1個です。

ここは今回のモデルと重要な違いがあります。

あなたの場合は**12疾患multi-label**です。

TransMILをそのまま使うと、

```text
1 CLS
 ↓
Linear
 ↓
12 logits
```

ですが、私は今回ならむしろ

```text
ACL token
PCL token
medial meniscus token
...
fracture token
```

の**12 target tokens**を持たせた方が自然だと思います。

---

# 6. 第一Transformer層

入力

\[
H=[h_{\mathrm{CLS}},h_1,\ldots,h_n]
\]

にself-attentionします。

普通のattentionなら、

\[
Q=HW_Q,\qquad
K=HW_K,\qquad
V=HW_V
\]

として

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)
\]

そして

\[
H'=AV
\]

です。

重要なのは

\[
A\in\mathbb R^{n\times n}
\]

というところ。

例えば、

```text
          window
         1    2    3    4
window1 .8   .1   .05  .05
window2 .1   .5   .3   .1
window3 .0   .2   .7   .1
window4 .1   .1   .2   .6
```

のように、window-window interactionができます。

これがABMILにはない情報です。

---

# 7. ただし普通のTransformerでは重すぎる

WSIでは \(n=10,000\) patch程度です。

通常のself-attentionは

\[
O(n^2)
\]

なので、

\[
10^8
\]

規模のattention matrixになってしまいます。

そこでTransMILは **Nyström Attention** を使います。

---

# 8. Nyström attentionとは何か

すべての

\[
QK^\top
\]

を計算するかわりに、\(m\) 個のlandmarkを選びます。

概念的には、

```text
10,000 tokens
       ↓
   256 landmarks
       ↓
token ↔ landmark
landmark ↔ landmark
landmark ↔ token
```

という低ランク近似です。

論文ではこれにより、固定したlandmark数なら計算量を \(n\) にほぼ線形にできます。

公開コードではさらに具体的で、

```python
NystromAttention(
    dim=512,
    dim_head=64,
    heads=8,
    num_landmarks=256,
    pinv_iterations=6,
    residual=True,
    dropout=0.1,
)
```

です。

つまり、

- hidden dimension: 512
- 8 heads
- head dimension: 64
- landmarks: 256

です。

### 今回のRSNA KneeではNyströmは要らない可能性が高い

あなたの場合、

```text
slots × series × windows
```

を全部flattenしても、例えば100〜300 tokens程度でしょう。

\[
300^2=90,000
\]

なので普通のfull self-attentionで全然問題ありません。

したがって**TransMILを再現するためだけにNyströmを持ってくる必要はない**です。

むしろ普通の

```python
nn.MultiheadAttention
```

またはSDPA/Flash Attentionを使った方が簡単です。

---

# 9. TransMILの特徴的なPPEG

もう一つの主要な工夫が

**Pyramid Position Encoding Generator (PPEG)**

です。

第一Transformer層の後で、patch tokensを一度2次元に戻します。

そしてdepthwise convolutionで

\[
3\times3,\quad5\times5,\quad7\times7
\]

の3種類をかけます。

公開実装は非常に明快です。

```python
self.proj  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
self.proj1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
self.proj2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
```

そして

```python
x = (
    cnn_feat
    + self.proj(cnn_feat)
    + self.proj1(cnn_feat)
    + self.proj2(cnn_feat)
)
```

です。

つまり

\[
H_{\rm PPEG}
=
H
+
Conv_{3}(H)
+
Conv_{5}(H)
+
Conv_{7}(H).
\]

これは

> Transformerでglobal correlation  
> + convolutionでlocal spatial context

という構造です。

かなり合理的です。

---

# 10. 「Squaring」とは何か

PPEGをするためにはtoken列を

\[
H\times W
\]

にしないといけません。

ところがpatch数 \(n\) は毎回違います。

そこで、

\[
H=W=\lceil\sqrt n\rceil
\]

として、

\[
H\times W
\]

個になるまで先頭のtokenをコピーします。

実装では、

```python
_H = _W = int(np.ceil(np.sqrt(H)))
add_length = _H * _W - H

h = torch.cat(
    [h, h[:, :add_length, :]],
    dim=1
)
```

です。

これを論文では **squaring of sequence** と呼びます。

ここは論文Algorithm 2の数式表記がやや紛らわしく、実装を見ると「平方数になるまでpaddingする」という意味が明確です。

---

# 11. ただしPPEGには重要な注意点がある

これはRSNA Kneeではかなり重要です。

公開コードのPPEGには、

```text
実際のWSI x座標
実際のWSI y座標
```

は入力されません。

単に**featureの並び順をpseudo-2D gridにreshapeしています**。

したがって、

> PPEG = 厳密な物理座標encoding

ではありません。

token orderingに空間構造が保存されていることに依存します。

これは今回のMRIではそのまま真似しない方がいいです。

あなたにはすでに

```python
positions
```

があります。

MRIなら、

\[
z_{\rm normalized},
\quad
\text{plane},
\quad
\text{series position}
\]

などの**本当の位置情報を使える**からです。

---

# 12. 第二Transformer層

PPEGで位置情報を入れたあと、もう一回self-attentionします。

```text
Transformer 1
 ↓
instance correlation

PPEG
 ↓
local/spatial information

Transformer 2
 ↓
position-aware global aggregation
```

という構成です。

つまりTransformerを大量にstackしたモデルではありません。

**たった2層**です。

これはMIL headとして使うには重要なポイントです。

---

# 13. 実装を見ると「普通のTransformer Encoder」とも少し違う

ここは論文を読むだけだと見落としやすいところです。

公開コードの `TransLayer` は、

```python
def forward(self, x):
    x = x + self.attn(self.norm(x))
    return x
```

だけです。

つまり典型的Transformerの

```text
Self Attention
   ↓
FFN / MLP
```

の**FFNがありません**。

典型的には

\[
x'=x+\operatorname{MHA}(\mathrm{LN}(x))
\]

の後に

\[
x''=x'+\operatorname{MLP}(\mathrm{LN}(x'))
\]

がありますが、公開TransMILでは後者がありません。

なので実態としては

> 2 × Nyström self-attention block + PPEG

くらいに考えた方が正確です。

---

# 14. 学習設定

論文では、

- batch size 1
- Cross Entropy
- Lookahead optimizer
- learning rate \(2\times10^{-4}\)
- weight decay \(10^{-5}\)
- patch embedding 1024 → 512
- ImageNet pretrained ResNet50

です。

公開configでは200 epochs、gradient accumulation 2、fp16を使っています。

---

# 15. 結果はどれくらい良かったか

一番面白いのはCAMELYON16です。

| Model | CAMELYON16 AUC |
|---|---:|
| ABMIL | 0.8760 |
| CLAM-SB | 0.8809 |
| CLAM-MB | 0.8679 |
| DSMIL | 0.8179 |
| **TransMIL** | **0.9309** |

TCGA-NSCLCでもTransMILはAUC 0.9603、TCGA-RCCでは0.9882でした。

特にCAMELYON16では陽性slideでも腫瘍領域が平均10%未満とされており、**少数の陽性instanceを大量の陰性instanceから拾うMILらしい条件**で差が大きかった点は興味深いです。

これは今回の

> 何十windowの中の一部だけがACL tearを示す

という状況にも近いところがあります。

---

# 16. PPEGのablationもかなり大きい

CAMELYON16 AUCでは、

\[
0.8416 \quad \text{position encodingなし}
\]

に対して、

\[
0.9309 \quad \text{PPEG}
\]

でした。

NSCLCでも

\[
0.9287\rightarrow0.9603
\]

です。

つまり論文の結果だけを見ると、

**Transformerそのものだけでなくposition/local structureがかなり重要**

ということになります。

---

# 17. 理論部分は少し割り引いて読むべき

論文では、

\[
H(\Theta_1,\ldots,\Theta_n)
\leq
\sum_i H(\Theta_i)
\]

という情報エントロピーの性質を使い、

> correlationを考慮すればuncertaintyを減らせる

と論じています。

式自体はconditional entropyの

\[
H(X|Y)\leq H(X)
\]

から来る標準的な関係です。

ただし、

> entropyが小さい  
> ⇒ Transformer MILの分類性能が必ず良い

ことを証明しているわけではありません。

したがって理論的な部分は、

**「instance間相関をMILへ導入する動機づけ」**

程度に読むのが妥当です。

また、

> 従来MILはi.i.d. assumption

という表現もやや強いです。

ABMILが数学的にinstance independenceを必須としているというより、

**pooling関数がinstance-instance interactionを明示的にモデル化していない**

と捉える方が実装上は正確です。

---

# 18. 「TransMILが勝った = self-attentionが原因」とまでは言えない

ここも重要です。

TransMILとABMILの差は、

```text
self-attention
+ CLS aggregation
+ Nyström approximation
+ PPEG
+ positional information
```

を全部まとめた差です。

PPEG単独のablationはありますが、

```text
完全に同一条件
ABMIL pooling vs full self-attention
```

というきれいな要因分解は十分ではありません。

したがって、

> 「instance correlationが5%以上AUCを上げた」

とは直接言えません。

これは今回、自分たちでablationする価値があります。

---

# 19. 今回のRSNA Kneeへの対応関係

ここが一番重要です。

今のモデルは、

```text
DINOv2
 ↓
window features

window attention
 ↓
series

series attention
 ↓
slot

slot attention
 ↓
study

target-specific classifier
```

です。

TransMIL思想にすると、

```text
DINOv2
 ↓
window tokens
 ↓
Transformer
 ↓
contextualized window tokens
 ↓
target-specific pooling
 ↓
12 logits
```

となります。

私は**完全なTransMILコピーではなく、この形が第一候補**だと思います。

特に、

\[
x_i
\rightarrow
\tilde x_i=
\mathrm{Transformer}(x_1,\dots,x_N)_i
\]

とまずwindowを文脈化し、その後は現在の疾患別attentionを残す。

つまり、

```text
             DINOv2
                ↓
       window embeddings
                ↓
       2-layer Transformer
                ↓
    contextualized embeddings
                ↓
 ┌──────────────┼─────────────┐
ACL attention  PCL attention  ... fracture attention
    ↓             ↓                   ↓
 logit          logit               logit
```

です。

これは

**TransMILのinstance correlation**

と

**現在のtarget-specific attention**

のいいところを両方残せます。

---

# 20. さらに一歩進めるなら「12 target token」

よりTransformerらしくするなら、

```text
ACL query
PCL query
MCL query
...
fracture query

        +

window tokens
```

を一緒にTransformerへ入れます。

すると、

\[
q_t'
=
\operatorname{Transformer}
(q_t,x_1,\ldots,x_N)
\]

となり、

\[
\hat y_t=w_t^\top q_t'
\]

とできます。

これは元TransMILの

```text
1 CLS token
```

を

```text
12 disease tokens
```

に拡張したものです。

今回はこちらの方が、元論文よりタスク構造に合っています。

---

## 今回TransMILから取り入れるもの・捨てるもの

私ならこうします。

| TransMIL要素 | RSNA Knee |
|---|---|
| CNN/DINO feature → MIL | **採用** |
| self-attentionによるinstance interaction | **採用** |
| 2層程度の浅いTransformer | **採用** |
| CLS token 1個 | 12 target tokenへ変更候補 |
| Nyström Attention | **不要** |
| pseudo-square化 | **不要** |
| PPEG | そのままは不要 |
| 位置情報 | **実MRI座標encodingで採用** |
| WSI patch間2D関係 | slice/series/plane関係に置換 |

要するに、**TransMILから本当に持ってくるべき核心はNyströmでもPPEGでもなく、「MIL poolingする前にinstancesをself-attentionで相互作用させる」という発想**です。

そして今回のモデルなら、

\[
\boxed{
\text{DINOv2}
\rightarrow
\text{Transformerによるwindow間相互作用}
\rightarrow
\text{疾患別attention}
\rightarrow
12\text{ logits}
}
\]

をまず試すのが、現在のattention MILに対する最もきれいな比較実験だと思います。

[TransMIL NeurIPS 2021 論文](https://proceedings.neurips.cc/paper/2021/file/10c272d06794d3e5785d5e7c5356e9ff-Paper.pdf?utm_source=chatgpt.com)  
[TransMIL公式実装](https://github.com/szc19990412/TransMIL?utm_source=chatgpt.com)