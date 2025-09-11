# SPEECH-MARC
An open-source speech-based assessment framework for detecting Mild Cognitive Impairment (MCI) using multimodal linguistic and acoustic features.

## 1. Preprocessing: Transcribe + Merge Demographics

SPEECH-MARC provides preprocessing tools to convert raw audio into transcripts, 
then merge them with participant demographics.

### Requirements
- [ffmpeg](https://ffmpeg.org/download.html) must be installed and accessible in your system PATH
  (needed for audio duration extraction).
- Python dependencies are handled automatically when you install the package.

### Example usage

```python
from pathlib import Path
from speechmarc.preprocessing import (
    TranscribeConfig,
    transcribe_folder,
    merge_transcripts_with_demographics,
)

# 1) Transcribe a folder of audio files
cfg = TranscribeConfig(
    audio_dir=Path("path/to/audio_files"),           # folder with .wav/.mp3 files
    out_txt_dir=Path("outputs/transcripts_txt"),     # will save one .txt per participant
    out_csv_path=Path("outputs/cookie_transcripts.csv"),  # combined transcripts CSV
    whisper_model_name="base",       # change to "small" / "medium" / "large-v3" as needed
    whisper_compute_type="int8",     # or "float16" if GPU supports
    filename_id_len=6,               # 'VR0123...' -> 'VR0123'
)
df_transcripts = transcribe_folder(cfg)

# 2) Merge transcripts with demographics CSV
merged = merge_transcripts_with_demographics(
    transcripts_csv=Path("outputs/cookie_transcripts.csv"),
    demographics_csv=Path("path/to/demographics.csv"),
    out_merged_csv=Path("outputs/merged_cookie_transcripts.csv"),
)
