"""Translate transcript segments while preserving their timestamps."""

from .translate import translate


def translate_segments(segments: list[dict], direction: str) -> list[dict]:
    return [
        {"start": segment["start"], "end": segment["end"],
         "text": translate(segment["text"], direction)}
        for segment in segments
    ]
