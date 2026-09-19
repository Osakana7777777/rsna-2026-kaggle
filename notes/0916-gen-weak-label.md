# Task: RSNA Knee Report Weak Label Extraction Pipeline

RSNA Knee Abnormality Detection competition用に、radiology reportから12疾患のweak label生成に必要な構造化情報を抽出するバッチ推論パイプラインを実装してください。

## 目的

約4,349件のradiology reportをLLMで解析し、competition-specific criteriaに沿った構造化情報を抽出する。

LLMには最終的な probability `p` や confidence weight `w` を直接出させない。

処理は以下の3段階に分離する。

```text
Radiology Report
        ↓
LLM structured extraction
        ↓
rule-based competition mapping
        ↓
p / confidence weight / mask
```

今回実装する主対象は、

1. vLLMによるLLM structured extraction
2. JSON validation
3. 結果保存
4. Python側で扱いやすいdecoder / post-processing基盤

とする。

---

# 1. 使用モデル

使用するモデル：

```text
Qwen/Qwen3.8-27B-FP8
```

H100 80GB 1枚での実行を優先してFP8を使用する。

推論は原則：

```text
thinking / reasoning = OFF
```

とする。

今回のタスクは自由推論ではなく、

```text
report evidence
→ predefined competition rubric
→ structured extraction
```

なので、通常症例でreasoning tokenを生成しない。

将来的に、

- threshold付近
- conflicting evidence
- unclear negation
- unclear acuity
- traumatic vs degenerativeが不明

などの症例だけ第二段階でreasoningを使える設計にはしてよいが、今回の本番first-passでは実装必須ではない。

---

# 2. vLLM設定

H100 80GB × 1を想定。

基本設定：

```text
enable_prefix_caching = True
structured output = JSON schema
thinking = OFF
```

`max_model_len` はまず

```text
65536
```

を使用する。

128KはQwen3.8側で対応可能なら設定可能にしてよいが、今回のreport extractionでは通常不要なので、CLI/configで変更可能なoptionとして扱う。

例えば：

```text
--max-model-len 65536
```

defaultとする。

可能なら：

```text
--kv-cache-dtype fp8
--gpu-memory-utilization 0.90〜0.95
```

も設定可能にする。

OOMしない範囲でbatch throughputを最大化する。

---

# 3. Prefix cachingを最大限効かせる

全症例で完全に同一のsystem promptを使用する。

可変部分は必ずprompt末尾のuser messageだけに置く。

構造：

```text
SYSTEM
├── fixed extraction instruction
├── fixed target definitions
├── fixed competition criteria
├── fixed code definitions
└── fixed output rules

USER
├── study UID
└── radiology report
```

以下のように、UIDやreportをsystem promptの途中に入れない。

BAD:

```text
UID
system prompt
report
rubric
```

GOOD:

```text
long fixed system prompt
↓
UID + report
```

---

# 4. 重要な設計原則

LLMには以下を出させない。

```text
probability
confidence weight
mask
final training label probability
```

これらは後でPython側で決定する。

LLMの役割は、

```text
「reportに何が書かれているか」
+
「competition thresholdに対してどこに位置するか」
```

を構造化すること。

特に以下を区別すること。

```text
explicit positive
explicit negative
equivocal
not mentioned
```

重要：

```text
not mentioned != negative
```

---

# 5. Target順序

JSONの `x` は必ず12要素で、以下の固定順序。

```python
TARGETS = [
    "ACL",
    "MCL",
    "Medial_Meniscus",
    "Lateral_Meniscus",
    "Medial_OA",
    "Lateral_OA",
    "PF_OA",
    "Effusion",
    "Synovitis",
    "Bakers",
    "Contusion",
    "Fracture",
]
```

target名をJSON内で毎回出力しない。

固定位置で解釈する。

---

# 6. 本番用compact output format

モデル出力：

```json
{
  "l": "en",
  "s": 2,
  "g": [],
  "x": [
    [1,1,12,1,3,[1,3],[],["Complete ACL rupture"]],
    ...
  ]
}
```

各findingは固定長8要素：

```text
[m, c, v, a, t, k, r, e]
```

意味：

```text
m = mention
c = certainty
v = severity
a = acuity
t = competition threshold relation
k = supported criterion feature IDs
r = review reason IDs
e = short verbatim evidence excerpts
```

---

# 7. Encoding

## mention

```python
MENTION = {
    0: "not_mentioned",
    1: "explicit_positive",
    2: "explicit_negative",
    3: "equivocal",
}
```

## certainty

```python
CERTAINTY = {
    0: "not_applicable",
    1: "definite",
    2: "probable",
    3: "possible",
    4: "equivocal",
}
```

## severity

```python
SEVERITY = {
    0: "not_stated_or_not_applicable",
    1: "none",
    2: "trace",
    3: "minimal",
    4: "small",
    5: "mild",
    6: "low_grade",
    7: "moderate",
    8: "high_grade",
    9: "severe",
    10: "large",
    11: "massive",
    12: "complete",
}
```

## acuity

```python
ACUITY = {
    0: "not_applicable",
    1: "not_stated",
    2: "acute",
    3: "subacute",
    4: "chronic",
    5: "remote_or_healed",
    6: "indeterminate",
}
```

## competition threshold relation

```python
THRESHOLD = {
    0: "insufficient_information",
    1: "clearly_below",
    2: "near_threshold",
    3: "clearly_above",
    4: "not_applicable",
}
```

---

# 8. Criterion feature IDs

```python
FEATURE = {
     1: "complete_tear",
     2: "high_grade_partial_tear",
     3: "fiber_discontinuity",
     4: "low_grade_sprain",
     5: "degenerative_ligament_change",

     6: "definite_meniscal_tear",
     7: "surface_reaching_meniscal_tear",
     8: "displaced_or_abnormal_meniscal_morphology",
     9: "intrameniscal_degeneration_only",

    10: "high_grade_cartilage_loss",
    11: "full_thickness_cartilage_loss",
    12: "large_cartilage_area",
    13: "low_grade_or_focal_chondrosis",

    14: "moderate_or_large_fluid_or_cyst",
    15: "trace_or_small_fluid_or_cyst",

    16: "synovitis_or_synovial_inflammation",
    17: "synovial_thickening_or_hypertrophy",

    18: "traumatic_bone_contusion",
    19: "nonspecific_marrow_edema",
    20: "degenerative_or_stress_marrow_edema",

    21: "acute_fracture_line",
    22: "acute_cortical_break",
    23: "chronic_or_healed_fracture",

    24: "explicitly_negated",
    25: "other_threshold_relevant_feature",

    26: "acute_periligamentous_edema",
    27: "osteophyte_or_sclerosis_without_qualifying_cartilage_loss",
    28: "fracture_associated_marrow_edema",
    29: "insufficiency_or_stress_lesion",
}
```

---

# 9. Review reason IDs

```python
REVIEW = {
    1: "severity_threshold_uncertain",
    2: "acuity_uncertain",
    3: "etiology_uncertain",
    4: "negation_uncertain",
    5: "laterality_uncertain",
    6: "conflicting_evidence",
    7: "insufficient_detail",
    8: "translation_uncertain",
    9: "other",
}
```

`needs_review`はモデルに出力させない。

Python側で

```python
needs_review = len(r) > 0
```

とする。

---

# 10. Global fields

```text
l = report language
```

ISO-639-1相当を優先。

不明：

```text
"und"
```

`s`:

```python
SIDE = {
    0: "unknown_or_not_stated",
    1: "left",
    2: "right",
    3: "ambiguous",
}
```

`g` global issues:

```python
GLOBAL_FLAGS = {
    1: "multiple_knees_mentioned",
    2: "target_knee_uncertain",
    3: "internally_conflicting_report",
    4: "possible_language_or_negation_issue",
    5: "insufficient_or_invalid_report",
}
```

---

# 11. Competition-specific extraction rules

System prompt内に以下のルールを必ず含める。

## ACL

competition positive:

```text
high-grade partial tear
complete tear
near-complete tear
clear >50% fiber disruption
clear discontinuity
```

competition negative / below threshold:

```text
low-grade sprain
degeneration
mucoid degeneration
thickening
mild signal abnormality
```

severity不明の`partial tear`を自動的にpositiveにしない。

---

## MCL

positive:

```text
acute high-grade partial tear
complete tear
clear fiber disruption
通常はacute edemaを伴う
```

below threshold:

```text
grade 1 / low-grade sprain
chronic thickening
remote injury
chronic stress-related change
```

単なる`MCL sprain`をpositiveにしない。

---

## Medial / Lateral Meniscus

positive:

```text
definite tear
surface-reaching tear
displaced fragment
truncated morphology
bucket-handle tear
radial tear
horizontal tear
longitudinal tear
complex tear
root tear
```

below threshold:

```text
intrasubstance degeneration
grade 1/2 signal
degenerative signal without definite tear
```

---

## OA

対象：

```text
Medial
Lateral
Patellofemoral
```

positive threshold:

```text
>50% cartilage thickness loss
AND
moderate/large area, approximately >= 1 cm
```

strong positive examples:

```text
full-thickness cartilage loss
extensive high-grade cartilage loss
large-area severe chondral loss
```

below threshold:

```text
mild chondrosis
superficial irregularity
focal low-grade cartilage loss
small isolated defect
```

以下だけではpositiveにしない：

```text
osteophytes
sclerosis
marrow edema/change
```

reportに単に`osteoarthritis`としか書かれていない場合、cartilage severity/extentを捏造しない。

---

## Effusion

positive:

```text
moderate
large
massive
```

below threshold:

```text
trace
minimal
slight
small
```

量が不明なら：

```text
insufficient_information
or near_threshold
```

---

## Synovitis

positive evidence:

```text
definite synovitis
synovial inflammation
inflammatory synovial thickening
synovial hypertrophy/proliferation interpreted as synovitis
```

禁止：

```text
effusion alone -> synovitis
```

nonspecific/reactive/equivocalならuncertainのままにする。

---

## Baker cyst

positive:

```text
moderate
large
```

below:

```text
trace
tiny
minimal
small
```

size不明なら自動positiveにしない。

---

## Contusion

positive:

```text
traumatic bone contusion
bone bruise
impact-related marrow edema
```

ただしdiscrete fracture lineがある場合は単純なcontusionにしない。

以下を自動positiveにしない：

```text
nonspecific marrow edema
degenerative marrow edema
OA-associated edema
stress-related edema
insufficiency-related edema
fracture-associated edema
```

---

## Fracture

positive:

```text
acute fracture line
acute cortical break
definite acute fracture
```

below:

```text
old fracture
healed fracture
remote fracture
chronic post-traumatic deformity
```

以下を自動positiveにしない：

```text
osteochondral abnormality
subchondral abnormality
stress/insufficiency abnormality without definite acute fracture line
```

---

# 12. Evidence extraction rules

`e`は原文のverbatim quote。

翻訳しない。

通常：

```text
最大1個
```

conflicting evidenceなど必要時のみ：

```text
最大2個
```

1 quoteはできる限り短くする。

`not mentioned`なら：

```json
[]
```

全文や長文をコピーしない。

---

# 13. Production JSON Schema

以下をstructured output schemaとして使用する。

```python
FINDING_SCHEMA = {
    "type": "array",
    "minItems": 8,
    "maxItems": 8,
    "prefixItems": [
        {
            "type": "integer",
            "enum": [0, 1, 2, 3],
        },
        {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
        },
        {
            "type": "integer",
            "minimum": 0,
            "maximum": 12,
        },
        {
            "type": "integer",
            "minimum": 0,
            "maximum": 6,
        },
        {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
        },
        {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 5,
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": 29,
            },
        },
        {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 4,
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": 9,
            },
        },
        {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "string",
                "maxLength": 160,
            },
        },
    ],
    "items": False,
}


RSNA_LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["l", "s", "g", "x"],
    "properties": {
        "l": {
            "type": "string",
            "minLength": 2,
            "maxLength": 8,
        },
        "s": {
            "type": "integer",
            "enum": [0, 1, 2, 3],
        },
        "g": {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 5,
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "x": {
            "type": "array",
            "minItems": 12,
            "maxItems": 12,
            "items": FINDING_SCHEMA,
        },
    },
}
```

もし使用しているvLLM / guided decoding backendでDraft 2020-12の`prefixItems`対応に問題がある場合は、同じ意味を保ったまま、そのvLLM versionで確実にvalidationできるschemaへ書き換えること。

schema validationを無効化することで回避しない。

---

# 14. User prompt

全症例で以下だけを変更する。

```text
StudyInstanceUID:
{study_uid}

Radiology report:
{report}
```

UIDはモデル出力に含めない。

Python側で入力UIDと出力を対応付ける。

---

# 15. System prompt

上記

- extraction principles
- encoding definitions
- competition-specific rules
- evidence rules
- output requirements

をすべて含む固定SYSTEM_PROMPTをコード内または外部テキストファイルとして保持する。

可能なら、

```text
prompts/rsna_knee_extractor_v1.txt
```

としてversion管理する。

prompt versionもmetadataに保存する。

---

# 16. 出力保存

raw extractionを必ず保存する。

推奨：

```text
outputs/
  raw_predictions.jsonl
```

1行：

```json
{
  "study_id": "...",
  "prompt_version": "v1",
  "model": "Qwen/Qwen3.8-27B",
  "prediction": {
    "l": "...",
    "s": 0,
    "g": [],
    "x": [...]
  }
}
```

さらにflat dataframeも生成する。

例えば：

```text
StudyInstanceUID
ACL_m
ACL_c
ACL_v
ACL_a
ACL_t
ACL_k
ACL_r
ACL_e
...
Fracture_m
...
```

Parquetで保存する。

```text
outputs/extracted_labels.parquet
```

JSONLをcanonical raw data、Parquetをanalysis用とする。

---

# 17. Checkpoint / resume

4,000件以上処理するので途中再開可能にする。

すでにJSONLに存在するStudyInstanceUIDはskipできるようにする。

例：

```bash
python infer.py \
    --input reports.csv \
    --output outputs/raw_predictions.jsonl \
    --resume
```

推論途中で失敗しても最初からやり直さない。

---

# 18. Validation

各predictionについて必ず以下を確認する。

```text
x length == 12
finding length == 8
enum range valid
e length <= 2
k / r IDs valid
JSON schema valid
```

schema-invalidなoutputは通常発生しない設計にする。

発生した場合は、

```text
StudyInstanceUID
raw output
error
```

をerror logに保存する。

silent failureは禁止。

---

# 19. Semantic sanity checks

JSON schemaだけでは検出できない矛盾も後処理でflagする。

例えば：

```text
m == not_mentioned AND e != []
```

は異常。

```text
m == explicit_negative AND feature includes complete_tear
```

も異常。

```text
t == clearly_above AND m == not_mentioned
```

も異常。

こうしたケースは修正せず、

```text
semantic_warning = true
```

として別途記録する。

---

# 20. competition_callはPythonで生成

モデルには生成させない。

```python
def competition_call(f):
    m, c, v, a, t, k, r, e = f

    if m == 0:
        return "uncertain"

    if m == 2:
        return "negative"

    if m == 3:
        return "uncertain"

    if t == 3:
        return "positive"

    if t == 1:
        return "negative"

    return "uncertain"
```

これによりLLMが

```text
finding exists
but below competition threshold
```

とした場合でも正しくcompetition-negativeにできる。

---

# 21. Initial confidence weight

最終値ではなく、後で58 gold labelsを使ってcalibrateする前提。

初期実装：

```python
def initial_weight(f):
    m, c, v, a, t, k, r, e = f

    if m == 0:
        w = 0.20
    elif c == 1 and t in (1, 3):
        w = 1.00
    elif c == 2:
        w = 0.70
    elif c == 3:
        w = 0.45
    else:
        w = 0.35

    if 6 in r:
        w *= 0.5

    if 4 in r:
        w *= 0.6

    if 5 in r:
        w *= 0.5

    if 8 in r:
        w *= 0.7

    if t == 2:
        w *= 0.7

    return max(0.05, min(1.0, w))
```

重要：

```text
w != 2 * abs(p - 0.5)
```

probabilityとconfidenceを分離する。

---

# 22. probabilityは別moduleにする

`p`のmappingは後から何度も変更するので、LLM推論コードに埋め込まない。

例えば：

```text
src/
  infer.py
  schema.py
  codes.py
  decode.py
  postprocess.py
  calibration.py
```

`calibration.py`側で、

```text
structured extraction
→ soft target probability
```

を扱う。

これによりLLMを再実行せず、

- Effusion threshold
- Baker cyst threshold
- unknown partial ACL tear
- Synovitis not-mentioned
- OA severity mapping

などを変更できるようにする。

---

# 23. Mask

`mask`もLLMには出させない。

後処理で、

```text
extremely uncertain
invalid report
target laterality unresolved
```

などを必要に応じて0にできるようにする。

defaultは1。

---

# 24. 実装品質

必須：

- type hints
- dataclassまたは明確なtyped structure
- logging
- deterministic configuration where possible
- configuration via CLI / YAML
- resume support
- unit tests
- dry-run mode

unit testには最低限以下を含める。

### Case 1

```text
Complete ACL rupture.
```

期待：

```text
ACL m=1
t=3
feature contains 1
```

### Case 2

```text
Small joint effusion.
```

期待：

```text
Effusion m=1
v=4
t=1
feature contains 15
```

### Case 3

```text
Moderate joint effusion.
```

期待：

```text
Effusion m=1
v=7
t=3
feature contains 14
```

### Case 4

```text
Mild degenerative signal within the medial meniscus without definite tear.
```

期待：

```text
Medial_Meniscus != competition positive
feature contains 9
```

### Case 5

```text
Bone marrow edema related to osteoarthritis.
```

期待：

```text
Contusion != competition positive
feature contains 20
```

### Case 6

```text
No acute fracture.
```

期待：

```text
Fracture m=2
feature contains 24
```

### Case 7

所見についてreportに記載なし。

期待：

```text
m=0
e=[]
```

---

# 25. Performance logging

以下を記録する。

```text
number of reports
prompt tokens
generated tokens
elapsed time
reports/sec
input tokens/sec
output tokens/sec
prefix cache hit metrics if obtainable
GPU memory usage
number of failures
number of semantic warnings
```

prefix cachingが実際に有効か確認できるログ・metricsを可能な範囲で残す。

---

# 26. Generation settings

structured extractionなのでcreative samplingは不要。

原則：

```text
temperature = 0
```

またはQwen/vLLMで推奨されるdeterministicな設定。

最大生成tokenは、12 findingsのcompact JSONに十分だが過剰でない値にする。

まず：

```text
max_tokens ≈ 1000–1500
```

程度を候補とし、実測後に縮める。

正常症例の出力token数分布を計測して、必要以上に大きく設定しない。

---

# 27. 将来のsecond-passに備える

今回必須ではないが、以下を抽出可能にしておく。

```python
def needs_second_pass(prediction):
    ...
```

候補：

```text
any r != []
any t == 2
global flag 3 or 4
semantic warning
```

このsubsetだけ将来的にQwen3.8のreasoning modeで再評価できるよう、first-passのコードとpost-processingを疎結合にする。

ただしfirst-passではreasoning OFF。

---

# 28. 最終的に欲しい実行形

例えば：

```bash
python -m src.infer \
  --input data/train_reports.csv \
  --report-col report \
  --id-col StudyInstanceUID \
  --model Qwen/Qwen3.8-27B \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --output outputs/raw_predictions.jsonl \
  --resume
```

推論後：

```bash
python -m src.postprocess \
  --input outputs/raw_predictions.jsonl \
  --output outputs/extracted_labels.parquet
```

将来的に：

```bash
python -m src.calibration \
  --features outputs/extracted_labels.parquet \
  --gold data/gold_labels.csv
```

と分離できるようにする。

---

# Acceptance criteria

以下を満たせば完成。

1. Qwen3.8-27BをvLLMでH100 1枚上でbatch inferenceできる。
2. prefix cachingが有効。
3. thinking/reasoning OFF。
4. 本番compact JSON schemaをstructured outputとして強制する。
5. 全reportで12 targetsを固定順序で出力する。
6. probability/confidence/maskはモデルに生成させない。
7. raw JSONLを保存する。
8. flattened Parquetを生成する。
9. resumeできる。
10. malformed outputを検出・記録する。
11. semantic sanity checksを行う。
12. competition_callをPython側で生成する。
13. 初期confidence weight計算を別moduleに実装する。
14. probability calibrationを後から交換可能な構造にする。
15. prompt version / model / inference settingsをmetadataとして保存する。
16. 最低限のunit testsを実装する。