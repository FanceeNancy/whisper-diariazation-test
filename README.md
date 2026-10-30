# Whisper + pyannote diarization app

This is the separate diarization app for a Python 3.12 environment.

## Setup

```bash
cd "/Users/nancymelchert/Desktop/whisper-diarization-app"
source .venv-diarization/bin/activate
python -V
```

Install FFmpeg if it is missing:

```bash
brew install ffmpeg
```

Then run:

```bash
streamlit run app.py
```

## Notes

- Use a Hugging Face token with access to `pyannote/speaker-diarization-3.1`.
- This app is intentionally isolated from the transcription-only app in the main project folder.
