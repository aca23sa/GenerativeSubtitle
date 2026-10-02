"""Helsinki-NLP Marian translation models for English and Arabic."""

from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {"en-ar": "Helsinki-NLP/opus-mt-en-ar", "ar-en": "Helsinki-NLP/opus-mt-ar-en"}


@lru_cache(maxsize=2)
def load_model(direction: str):
    if direction not in MODELS:
        raise ValueError("direction must be 'en-ar' or 'ar-en'")
    name = MODELS[direction]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def translate(text: str, direction: str, max_new_tokens: int = 256) -> str:
    if not text.strip():
        return text
    tokenizer, model = load_model(direction)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()
