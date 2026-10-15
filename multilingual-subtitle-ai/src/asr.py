"""Whisper based English and Arabic speech recognition."""

from functools import lru_cache
from typing import Any

import torch
from transformers import pipeline

DEFAULT_MODEL = "openai/whisper-small"
LANGUAGES = {"english": "english", "en": "english", "arabic": "arabic", "ar": "arabic"}


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str) -> Any:
    # Transformers expects an integer device index (or -1 for CPU), not "cpu".
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_name, device=device)


def transcribe(audio_file: str, language: str,
               model_name: str = DEFAULT_MODEL) -> dict:
    """Transcribe audio with Whisper and return timestamped chunks."""
    try:
        whisper_language = LANGUAGES[language.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("language must be 'english'/'en' or 'arabic'/'ar'") from exc

    return _load_pipeline(model_name)(
        audio_file,
        generate_kwargs={"language": whisper_language, "task": "transcribe"},
        return_timestamps=True,
    )
