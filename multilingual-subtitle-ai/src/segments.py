"""Convert timestamped ASR output into the pipeline's segment format."""


def convert_segments(result: dict) -> list[dict]:
    segments = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp")
        text = chunk.get("text", "").strip()
        if not timestamp or len(timestamp) != 2:
            continue
        start, end = timestamp
        if start is None or end is None or not text:
            continue
        start, end = float(start), float(end)
        if start < 0 or end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text})

    # Keep a usable SRT when an ASR backend returns text without chunk timestamps.
    if not segments and result.get("text", "").strip():
        segments.append({"start": 0.0, "end": 1.0, "text": result["text"].strip()})
    return segments
