"""End-to-end local video to transcript/subtitle pipeline."""

from pathlib import Path
import tempfile

from .audio import extract_audio
from .asr import transcribe
from .segments import convert_segments
from .subtitles import create_srt
from .translation_pipeline import translate_segments


def generate_subtitles(video_file: str | Path, language: str,
                       translation_direction: str | None = None,
                       model_name: str = "openai/whisper-small") -> list[dict]:
    """Extract, transcribe and optionally translate a video, returning segments."""
    with tempfile.TemporaryDirectory(prefix="subtitle-ai-") as temp_dir:
        audio_file = Path(temp_dir) / "audio.wav"
        extract_audio(video_file, audio_file)
        segments = convert_segments(transcribe(str(audio_file), language, model_name))
    if translation_direction:
        segments = translate_segments(segments, translation_direction)
    return segments


def generate_subtitle_file(video_file: str | Path, output_file: str | Path,
                           language: str, translation_direction: str | None = None,
                           model_name: str = "openai/whisper-small") -> Path:
    segments = generate_subtitles(video_file, language, translation_direction, model_name)
    return create_srt(segments, output_file)
