import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

class SpeechTranscriptPreprocessor:
    """Cleans time-series speech transcriptions and tokenizes text for BERT/RoBERTa."""
    def __init__(self, model_name: str = "roberta-base", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Remove filler words or artifacts while retaining pause markers if needed
        cleaned = text.strip()
        return cleaned

    def tokenize_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        cleaned_texts = [self.clean_text(t) for t in texts]
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
