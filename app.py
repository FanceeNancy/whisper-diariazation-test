import csv
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import whisper
from pyannote.audio import Pipeline


MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff"}


def ensure_ffmpeg_on_path():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            os.environ["PATH"] = str(Path(candidate).parent) + os.pathsep + os.environ.get("PATH", "")
            return candidate
    return None


def normalize_audio_file(file_bytes, file_name: str, tmp_dir: str) -> str:
    original_path = os.path.join(tmp_dir, file_name)
    with open(original_path, "wb") as f:
        f.write(file_bytes)

    suffix = Path(file_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_name}")

    ffmpeg_path = ensure_ffmpeg_on_path()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is not installed or not on PATH. Install it with Homebrew: brew install ffmpeg")

    output_path = os.path.join(tmp_dir, f"{Path(file_name).stem}_normalized.wav")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        original_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to normalize audio: {result.stderr}")
    return output_path


@st.cache_resource
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name)


def get_hf_token() -> str | None:
    token = (st.session_state.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip()
    return token or None


def iter_diarized_tracks(diarization):
    candidates = []

    if hasattr(diarization, "speaker_diarization"):
        candidates.append(getattr(diarization, "speaker_diarization"))
    if hasattr(diarization, "exclusive_speaker_diarization"):
        candidates.append(getattr(diarization, "exclusive_speaker_diarization"))
    candidates.append(diarization)

    for candidate in candidates:
        iter_tracks = getattr(candidate, "iter_tracks", None)
        if callable(iter_tracks):
            iterator = iter_tracks(yield_label=True)
        else:
            itertracks = getattr(candidate, "itertracks", None)
            if not callable(itertracks):
                continue
            iterator = itertracks(yield_label=True)

        for item in iterator:
            if len(item) == 3:
                turn, _, speaker = item
            elif len(item) == 2:
                turn, speaker = item
            else:
                continue
            yield {"start": turn.start, "end": turn.end, "speaker": speaker}
        return

    raise AttributeError(f"{type(diarization).__name__} does not provide a supported diarized-track iterator")


def diarize_audio(audio_path: str, hf_token: str | None = None):
    token = hf_token or get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required. Paste your Hugging Face token in the sidebar or export HF_TOKEN before launching the app.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    diarization = pipeline(audio_path, min_speakers=1, max_speakers=5)

    return list(iter_diarized_tracks(diarization))


def merge_diarized_segments(segments, diarized_segments):
    speaker_map = {}
    for idx, segment in enumerate(segments):
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        best_speaker = f"Speaker {((idx % 2) + 1)}"
        best_overlap = 0.0

        for turn in diarized_segments:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        speaker_map[idx] = best_speaker
    return speaker_map


def format_timecode(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def build_srt(segments, speaker_map=None):
    speaker_map = speaker_map or {}
    srt_lines = []
    for idx, segment in enumerate(segments, start=1):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = speaker_map.get(idx - 1, speaker_map.get(idx, f"Speaker {((idx - 1) % 2) + 1}"))
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_timecode(segment['start'])} --> {format_timecode(segment['end'])}")
        srt_lines.append(f"{speaker}: {text}")
        srt_lines.append("")
    return "\n".join(srt_lines).strip() + "\n"


def build_text_transcript(segments, speaker_map=None):
    speaker_map = speaker_map or {}
    blocks = []
    for idx, segment in enumerate(segments):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = speaker_map.get(idx, f"Speaker {((idx % 2) + 1)}")
        blocks.append(f"[{int(segment['start']) // 60:02d}:{int(segment['start']) % 60:02d} - {int(segment['end']) // 60:02d}:{int(segment['end']) % 60:02d}] {speaker}: {text}")
    return "\n".join(blocks).strip() + "\n"


def build_csv(segments, speaker_map=None):
    speaker_map = speaker_map or {}
    rows = []
    for idx, segment in enumerate(segments):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = speaker_map.get(idx, f"Speaker {((idx % 2) + 1)}")
        rows.append(
            {
                "start_time": format_timecode(segment["start"]),
                "end_time": format_timecode(segment["end"]),
                "duration_seconds": f"{max(segment.get('end', 0.0) - segment.get('start', 0.0), 0.0):.3f}",
                "speaker": speaker,
                "text": text,
            }
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["start_time", "end_time", "duration_seconds", "speaker", "text"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_export_bundle(transcript_text: str, srt_text: str, csv_text: str, stem: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}_transcript.txt", transcript_text)
        archive.writestr(f"{stem}_transcript.srt", srt_text)
        archive.writestr(f"{stem}_transcript.csv", csv_text)
    return buffer.getvalue()


def transcribe_file(file_bytes, file_name: str, model_name: str, task: str, language: str | None, hf_token: str | None = None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_audio_path = normalize_audio_file(file_bytes, file_name, tmp_dir)
        model = load_whisper_model(model_name)

        transcription = model.transcribe(
            input_audio_path,
            fp16=False,
            task=task,
            language=language if language else None,
            verbose=False,
            word_timestamps=True,
        )

        segments = transcription.get("segments", [])
        speaker_map = {}
        if segments:
            try:
                diarized_turns = diarize_audio(input_audio_path, hf_token=hf_token)
                speaker_map = merge_diarized_segments(segments, diarized_turns)
            except Exception as exc:
                raise RuntimeError(f"Diarization failed: {exc}") from exc

        summary = {
            "language": transcription.get("language", "unknown"),
            "duration": transcription.get("segments", [{}])[-1].get("end", 0.0) if segments else 0.0,
            "segment_count": len(segments),
            "text": transcription.get("text", "").strip(),
            "speaker_count": len(set(speaker_map.values())) if speaker_map else 0,
        }

        transcript_text = build_text_transcript(segments, speaker_map=speaker_map)
        srt_text = build_srt(segments, speaker_map=speaker_map)
        csv_text = build_csv(segments, speaker_map=speaker_map)
        return summary, transcript_text, srt_text, csv_text


st.set_page_config(page_title="Speaker Diarization App", page_icon="🎙️", layout="wide")
st.title("Whisper + pyannote diarization")
st.caption("Separate diarization app for Python 3.12 + FFmpeg + Hugging Face access")


with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Whisper model", MODEL_OPTIONS, index=1)
    task = st.selectbox("Output mode", ["transcribe", "translate"], index=0)
    language = st.text_input("Language code (optional)", help="Examples: en, es, fr")
    language = language.strip() or None

    hf_token = st.text_input("Hugging Face token", type="password", help="Required for pyannote/speaker-diarization-3.1")
    st.session_state["hf_token"] = hf_token.strip() or st.session_state.get("hf_token", "")

uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=list(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS),
    accept_multiple_files=False,
    help="Maximum file size: 500 MB",
)

if uploaded_file is not None:
    if st.button("Transcribe file"):
        with st.spinner("Processing audio and diarizing speakers..."):
            try:
                summary, transcript_text, srt_text, csv_text = transcribe_file(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    model_name,
                    task,
                    language,
                    hf_token=hf_token.strip() or None,
                )

                st.success("Transcription complete")
                st.subheader("Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Detected language", summary["language"])
                col2.metric("Segments", summary["segment_count"])
                col3.metric("Duration", f"{summary['duration']:.1f}s")
                col4.metric("Speakers", summary["speaker_count"])

                st.subheader("Transcript preview")
                st.text_area("Full transcript", transcript_text, height=400)

                st.subheader("SRT export")
                st.code(srt_text, language="text")

                file_stem = Path(uploaded_file.name).stem
                export_bundle = build_export_bundle(transcript_text, srt_text, csv_text, file_stem)
                download_cols = st.columns(4)
                download_cols[0].download_button("Download transcript", data=transcript_text, file_name=f"{file_stem}_transcript.txt", mime="text/plain")
                download_cols[1].download_button("Download SRT", data=srt_text, file_name=f"{file_stem}_transcript.srt", mime="text/plain")
                download_cols[2].download_button("Download CSV", data=csv_text, file_name=f"{file_stem}_transcript.csv", mime="text/csv")
                download_cols[3].download_button("Download all exports", data=export_bundle, file_name=f"{file_stem}_exports.zip", mime="application/zip")
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
else:
    st.markdown(
        """
        ### Setup notes
        1. Confirm `ffmpeg` is installed and on PATH.
        2. Make sure your Hugging Face token has access to `pyannote/speaker-diarization-3.1`.
        3. Upload an audio or video file and click Transcribe file.

        If `ffmpeg` is missing, run:
        `brew install ffmpeg`
        """
    )
