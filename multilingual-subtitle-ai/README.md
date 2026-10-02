# English/Arabic subtitle prototype

This Phase 1 prototype converts a video to mono 16 kHz audio with FFmpeg, uses
pretrained Whisper to produce timestamped English or Arabic speech, optionally
translates each segment with a Helsinki-NLP Marian model, and writes an SRT.
Models are downloaded from Hugging Face on first use and cached locally.

## Setup

Python 3.10 or newer and FFmpeg must be installed. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the commands below from this directory with that virtual environment
activated. If you prefer not to activate it, replace `python` with
`.venv/bin/python` (Windows: `.venv\\Scripts\\python.exe`).

Check that `ffmpeg -version` works in the same terminal.

## Generate subtitles

Transcribe English speech:

```sh
python run.py data/raw/sampleaudio.mp4 --language english --output outputs/english.srt
```

Transcribe Arabic speech:

```sh
python run.py data/raw/arabic_video.mp4 --language arabic --output outputs/arabic.srt
```

Transcribe English and translate to Arabic:

```sh
python run.py movie.mp4 --language english --translate en-ar --output outputs/arabic.srt
```

For Arabic speech translated to English, use `--language arabic --translate ar-en`.
`--model` can select another compatible Whisper checkpoint; the default is
`openai/whisper-small`. The translation direction must match the spoken source
language. CUDA is used when available; otherwise inference runs on CPU.

The equivalent Python entry point is `src.pipeline.generate_subtitle_file`.
Each generated segment retains its ASR timestamps when translated. Subtitle
line breaking, reading speed adjustment, dataset preparation, and model
training are intentionally outside this baseline milestone.
