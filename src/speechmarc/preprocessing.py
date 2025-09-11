"""
Preprocessing utilities for SPEECH-MARC.

Includes:
- Transcription of audio files with faster-whisper
- Audio duration extraction via pydub/ffmpeg
- Saving one .txt transcript per recording + a combined CSV
- Lightweight text cleaning
- Merging transcripts with a demographics CSV on participant_id
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import os
import time
import pandas as pd

# Optional (only needed for transcription and duration)
try:
    from faster_whisper import WhisperModel
    from pydub.utils import mediainfo
except Exception:
    WhisperModel = None
    mediainfo = None


# -----------------------------
# Text utilities
# -----------------------------
def clean_text(s: str) -> str:
    """Very light cleaner: trim, collapse whitespace."""
    return " ".join(str(s).strip().split())


# -----------------------------
# Configuration for transcription
# -----------------------------
@dataclass
class TranscribeConfig:
    audio_dir: Path
    out_txt_dir: Path
    out_csv_path: Path
    whisper_model_name: str = "base"      # e.g., "base", "small", "medium", "large-v3"
    whisper_compute_type: str = "int8"    # e.g., "int8", "float16", "int8_float16"
    file_globs: tuple = (".wav", ".mp3", ".m4a", ".mp4")
    filename_id_len: int = 6              # e.g., "VR0123..." -> "VR0123"
    overwrite_txt: bool = False           # if False, skip transcripts that already exist


# -----------------------------
# Core helpers
# -----------------------------
def _get_audio_duration_seconds(file_path: Path) -> Optional[float]:
    """Return duration in seconds using pydub.mediainfo, or None if unavailable."""
    if mediainfo is None:
        return None
    try:
        info = mediainfo(str(file_path))
        if "duration" in info:
            return round(float(info["duration"]), 2)
    except Exception:
        pass
    return None


def _transcribe_file(model: WhisperModel, audio_path: Path) -> str:
    """Run faster-whisper on a single file and return a single string transcript."""
    segments, _info = model.transcribe(str(audio_path))
    return " ".join(seg.text.strip() for seg in segments)


# -----------------------------
# Public API
# -----------------------------
def transcribe_folder(cfg: TranscribeConfig) -> pd.DataFrame:
    """
    Transcribe all matching audio files in cfg.audio_dir.
    Writes:
      - one .txt per recording into cfg.out_txt_dir
      - a combined CSV of transcripts into cfg.out_csv_path
    Returns the combined DataFrame.
    """
    if WhisperModel is None:
        raise ImportError("faster_whisper is not installed. Add it to your environment.")

    cfg.out_txt_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    model = WhisperModel(cfg.whisper_model_name, compute_type=cfg.whisper_compute_type)
    start = time.time()

    for filename in sorted(os.listdir(cfg.audio_dir)):
        if not filename.endswith(cfg.file_globs):
            continue
        # participant_id from prefix, e.g., 'VR0123...' -> 'VR0123'
        participant_id = filename[: cfg.filename_id_len]
        audio_path = cfg.audio_dir / filename
        txt_path = cfg.out_txt_dir / f"{participant_id}.txt"

        if txt_path.exists() and not cfg.overwrite_txt:
            # Skip if we already have a transcript for this participant
            # (use overwrite_txt=True to regenerate)
            transcript_text = txt_path.read_text(encoding="utf-8").strip()
        else:
            transcript_text = _transcribe_file(model, audio_path)
            txt_path.write_text(transcript_text, encoding="utf-8")

        duration_sec = _get_audio_duration_seconds(audio_path)

        records.append(
            {
                "participant_id": participant_id,
                "transcript_raw": transcript_text,
                "audio_duration_sec": duration_sec,
                "audio_filename": filename,
            }
        )

    df = pd.DataFrame(records)
    # (Optional) clean now or later:
    df["transcript_clean"] = df["transcript_raw"].map(clean_text)

    cfg.out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out_csv_path, index=False)

    elapsed = round(time.time() - start, 2)
    print(f"Transcribed {len(df)} files in {elapsed} sec")
    print(f"Saved transcripts CSV → {cfg.out_csv_path}")
    print(f"Saved per-file .txt → {cfg.out_txt_dir}")

    return df


def merge_transcripts_with_demographics(
    transcripts_csv: Path,
    demographics_csv: Path,
    out_merged_csv: Path,
    id_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Merge transcripts CSV with demographics CSV on participant_id.
    Ensures id columns are strings with whitespace stripped.
    """
    tdf = pd.read_csv(transcripts_csv)
    ddf = pd.read_csv(demographics_csv)

    # Normalize IDs
    tdf[id_col] = tdf[id_col].astype(str).str.strip()
    ddf[id_col] = ddf[id_col].astype(str).str.strip()

    merged = tdf.merge(ddf, on=id_col, how="left")

    # Report missing demos (sanity check)
    missing = merged.loc[merged.isna().any(axis=1), id_col].unique().tolist()
    if missing:
        print(f"⚠ Missing demographics for {len(missing)} participants (showing up to 20): {missing[:20]}")

    out_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_merged_csv, index=False)
    print(f"Saved merged CSV → {out_merged_csv}")
    return merged
