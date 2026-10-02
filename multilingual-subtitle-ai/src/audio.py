"""Video to mono, 16 kHz WAV conversion using the FFmpeg executable."""

import shutil
import subprocess
from pathlib import Path


def extract_audio(video_file: str | Path, output_file: str | Path) -> Path:
    source = Path(video_file).expanduser()
    destination = Path(output_file).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Install it and ensure 'ffmpeg' is on PATH.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Could not extract audio from {source}: {detail}") from exc
    return destination
