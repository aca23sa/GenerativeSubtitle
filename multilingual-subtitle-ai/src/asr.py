import torch
from transformers import pipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

model = "openai/whisper-small"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    device=device,
)


def transcribe(audio_file, language):
    result = pipe(
        audio_file,
        generate_kwargs={
            "language": language,
            "task": "transcribe"
        },
        return_timestamps=True
    )

    return result