import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class ConfusionEmotionClassifier(nn.Module):
    """Sequence classification wrapper for detecting confusion & emotion from text transcripts."""
    def __init__(self, model_name: str = "roberta-base", num_classes: int = 3):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
