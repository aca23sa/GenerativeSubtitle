"""SRT formatting and writing helpers."""

from pathlib import Path


def format_time(seconds: float) -> str:
    milliseconds_total = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def create_srt(segments: list[dict], output_file: str | Path) -> Path:
    destination = Path(output_file).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for index, segment in enumerate(segments, start=1):
            start, end = float(segment["start"]), float(segment["end"])
            text = str(segment["text"]).strip()
            if start < 0 or end <= start:
                raise ValueError(f"Segment {index} must have non-negative, increasing timestamps")
            if not text:
                continue
            file.write(f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n")
    return destination
