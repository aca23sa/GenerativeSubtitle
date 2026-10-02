"""Command line entry point: video -> timestamped SRT."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate English or Arabic subtitles from a video.")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--language", required=True, choices=("english", "arabic", "en", "ar"),
                        help="Spoken language in the video")
    parser.add_argument("--translate", choices=("en-ar", "ar-en"),
                        help="Optionally translate recognized speech")
    parser.add_argument("--output", help="Output SRT path (defaults beside video)")
    parser.add_argument("--model", default="openai/whisper-small", help="Hugging Face Whisper model")
    args = parser.parse_args()
    suffix = args.translate or args.language
    output = args.output or str(Path(args.video).with_suffix(f".{suffix}.srt"))
    # Import ML dependencies only after argument parsing so --help works even
    # when the user has not installed the model packages yet.
    try:
        from src.pipeline import generate_subtitle_file
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers", "sentencepiece"}:
            parser.error(
                f"missing dependency '{exc.name}'. Activate this project's virtual "
                "environment and run: python -m pip install -r requirements.txt"
            )
        raise
    path = generate_subtitle_file(args.video, output, args.language, args.translate, args.model)
    print(f"Wrote subtitles to {path}")


if __name__ == "__main__":
    main()
