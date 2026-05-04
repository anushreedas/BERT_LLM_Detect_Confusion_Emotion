# BERT-Based Confusion Emotion Detection in Speech Transcripts

A fine-tuned BERT classifier that detects multi-level human confusion from time-series transcribed speech data, with rigorous statistical significance testing to validate model comparisons.

This is the capstone project for **CSCI-739 Topics in Intelligent Systems** at Rochester Institute of Technology, involving a complete NLP pipeline from raw timestamped transcription data through to statistically validated model selection.

---

## Problem

Detecting confusion in spoken dialogue is a subtle and subjective task. A person's speech transcription may shift gradually across four confusion states (no confusion → mild → moderate → high), and the boundary between states is rarely clean. This project frames it as a sentence-level multi-class classification problem: given a sentence from a speech transcript, predict which of four confusion states the speaker was in when they said it.

The labels are encoded in the dataset using a custom base85 encoding scheme (`CONF_LABELS`) that maps participant and session identifiers to timestamped confusion state transitions. Decoding these produces a per-token label vector aligned to the transcript, which is then aggregated to sentence-level labels by taking the maximum confusion state observed within each sentence.

---

## Pipeline overview

```
Raw transcription CSVs (timestamped word tokens)
        ↓
Decode confusion labels from base85 → align to token timestamps
        ↓
Segment tokens into sentences → assign sentence-level label (max of token labels)
        ↓
Deduplicate sentences across participants
        ↓
Train / Val / Test split (stratified, 74% / 16% / 10%)
        ↓
Downsample majority classes → balanced training and validation sets
        ↓
Tokenize with BERT tokenizer → fine-tune BertClassifier
        ↓
Evaluate on 20 stratified test samples → sign test for statistical significance
```

---

## Models compared

Twelve model configurations were evaluated, varying two dimensions: the base model (BERT-base vs TOD-BERT) and the number of Transformer hidden layers (2, 6, 12, 14, 18):

| Configuration | Base model | Hidden layers | Frozen |
|---|---|---|---|
| bert-base-uncased_2 | BERT-base | 2 | No |
| bert-base-uncased_6 | BERT-base | 6 | No |
| bert-base-uncased_12 | BERT-base | 12 (full) | No |
| bert-base-uncased_14 | BERT-base | 14 | No |
| bert-base-uncased_18 | BERT-base | 18 | No |
| TOD-BERT_2 | TOD-BERT-JNT-V1 | 2 | No |
| TOD-BERT_6 | TOD-BERT-JNT-V1 | 6 | No |
| TOD-BERT_12 | TOD-BERT-JNT-V1 | 12 (full) | No |
| TOD-BERT_14 | TOD-BERT-JNT-V1 | 14 | No |
| TOD-BERT_18 | TOD-BERT-JNT-V1 | 18 | No |
| bert-base-uncased_12freeze | BERT-base | 12 | Yes (full freeze) |
| **bert-base-uncased_12freeze6** | **BERT-base** | **12** | **Partial (last 6 layers unfrozen)** |

**TOD-BERT** (`TODBERT/TOD-BERT-JNT-V1`) is a BERT variant pre-trained specifically on task-oriented dialogue datasets, making it a natural candidate for conversational confusion detection.

### Architecture

Each model wraps a BERT encoder with a two-layer feedforward classifier head:

```
BERT encoder ([CLS] token representation, dim 768)
        ↓
Linear(768 → 50) → ReLU → Linear(50 → 4)
        ↓
Softmax → confusion class {0, 1, 2, 3}
```

Training: Adam optimizer (lr=5e-5), linear LR warmup (200 steps), CrossEntropyLoss, 4 epochs, batch size 16.

---

## Results

Each model was evaluated on 20 stratified random samples drawn from the test set (30% of test data each, stratified by confusion class). This produces a distribution of 20 accuracy scores per model — the variance across samples captures uncertainty in the test evaluation.

**Mean accuracy across 20 test samples:**

| Model | Mean Acc | Min | Max |
|---|---|---|---|
| bert-base-uncased_2 | 0.762 | 0.719 | 0.787 |
| bert-base-uncased_6 | **0.817** | 0.787 | 0.835 |
| bert-base-uncased_12 | 0.800 | 0.777 | 0.833 |
| bert-base-uncased_14 | 0.808 | 0.777 | 0.843 |
| bert-base-uncased_18 | 0.797 | 0.772 | 0.820 |
| TOD-BERT_2 | 0.764 | 0.736 | 0.790 |
| TOD-BERT_6 | 0.800 | 0.777 | 0.825 |
| TOD-BERT_12 | 0.790 | 0.770 | 0.820 |
| TOD-BERT_14 | 0.789 | 0.759 | 0.808 |
| TOD-BERT_18 | 0.792 | 0.764 | 0.815 |
| bert-base-uncased_12freeze | 0.305 | 0.268 | 0.333 |
| **bert-base-uncased_12freeze6** | **0.814** | 0.790 | 0.841 |

The fully frozen BERT model (`12freeze`) collapsed to near-random performance (~0.30 for a 4-class problem), confirming that at least partial fine-tuning is necessary for this domain. Partially unfreezing the last 6 layers (`12freeze6`) recovered performance to match the best full fine-tuning configurations.

---

## Statistical significance testing

Comparing model accuracies on a single test split is insufficient — it doesn't tell you whether one model is *reliably* better or just got lucky on that particular sample. To address this, each model was evaluated on 20 independently drawn stratified samples, producing 20 accuracy scores per model.

The **sign test** (a non-parametric binomial test) was then used to determine whether one model's scores were consistently higher than another's:

```python
def sign_test(scores1, scores2):
    diff = [a - b for a, b in zip(scores1, scores2)]
    pos_count = sum(1 for d in diff if d > 0)
    neg_count = sum(1 for d in diff if d < 0)
    zero_count = sum(1 for d in diff if d == 0)
    
    # H0: model1 and model2 are equally good (p=0.5)
    p_value = binom_test(x=pos_count, n=len(scores1)-zero_count, p=0.5, alternative='greater')
    return p_value
```

Interpretation: p < 0.05 → model 1 is statistically significantly better than model 2; p < 0.10 → marginal evidence; p ≥ 0.10 → no significant difference.

The sign test was applied pairwise across all BERT-base configurations and separately across all TOD-BERT configurations.

---

## Data preprocessing details

**Label decoding.** Confusion labels are stored in `CONF_LABELS` as base85-encoded strings. Decoding produces a list of `(time_decisecond, confusion_state)` pairs marking when the confusion state changed during the recording. These are aligned to transcript tokens using their timestamps.

**Sentence segmentation.** Tokens are grouped into sentences by splitting on `.` and `?` punctuation tokens. The confusion label for each sentence is the maximum token-level label observed within it — a sentence is as confused as its most confused moment.

**Deduplication.** Sentences appearing in multiple participants' transcripts are deduplicated to prevent data leakage across train/val/test splits.

**Class balancing.** The confusion label distribution is heavily skewed toward class 0 (no confusion). Training and validation sets are downsampled to the size of the smallest class (class 3) to prevent the model from learning to predict "no confusion" for everything.

---

## Requirements

```
torch
transformers
datasets
scikit-learn
pandas
numpy
scipy
matplotlib
```

---

## Academic context

Capstone project for **CSCI-739 Topics in Intelligent Systems**, Rochester Institute of Technology, Spring 2021.

Demonstrates: NLP data preprocessing, custom label alignment from timestamped annotations, BERT fine-tuning and transfer learning, multi-class text classification, dataset balancing, and non-parametric statistical significance testing (sign test) for model comparison.

---

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://anushree-das.medium.com)
