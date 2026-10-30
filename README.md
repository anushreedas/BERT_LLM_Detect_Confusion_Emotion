# BERT & LLM Confusion and Emotion Detection in Transcribed Speech (`BERT_LLM_Detect_Confusion_Emotion`)

A production-grade, modular machine learning repository for detecting cognitive confusion and emotional states from time-series transcribed speech data using fine-tuned transformer models (BERT/RoBERTa) and LLM inference pipelines.

---

## Repository Architecture

```text
BERT_LLM_Detect_Confusion_Emotion/
├── assets/                          # Report figures, confusion matrices & plots
├── data/
│   ├── raw/                         # Raw transcribed audio transcripts & annotations
│   └── processed/                   # Tokenized tensors & train/val/test splits
├── notebooks/                       # Numbered exploratory & execution notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing/               # Text cleaning, tokenization & dataset wrapping
│   │   ├── __init__.py
│   │   └── text_tokenizer.py
│   ├── models/                      # Transformer classification & LLM prompting heads
│   │   ├── __init__.py
│   │   └── transformer_classifier.py
│   └── evaluation/                  # Classification metrics & error analysis
│       ├── __init__.py
│       └── metrics.py
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt
python notebooks/01_data_preprocessing.ipynb
python notebooks/02_model_training.ipynb
```
