# SPEECH-MARC
An open-source speech-based assessment framework for detecting Mild Cognitive Impairment (MCI) using multimodal linguistic and acoustic features.

## 1. Preprocessing: Transcribe + Merge Demographics

SPEECH-MARC provides preprocessing tools to convert raw audio into transcripts, 
then merge them with participant demographics.

### Requirements
- [ffmpeg](https://ffmpeg.org/download.html) must be installed and accessible in your system PATH
  (needed for audio duration extraction).
- [spacy] (https://github.com/explosion/spaCy) must be installed for nlp.
- [opensmile] (https://audeering.github.io/opensmile/get-started.html)
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

### Notes
- You must have **ffmpeg** installed for `mediainfo` to return audio durations.
- If participant IDs are not the first 6 characters of the filename, adjust `filename_id_len`.
- If filenames vary (e.g., `VR0123 Cookie Theft 2024.wav`), the prefix extraction 
  still works since only the first 6 characters are used.

## 2. Feature Extraction

SPEECH-MARC supports three types of feature extraction:

1. **Linguistic + ICU/AOI features** from transcript text  
2. **Acoustic eGeMAPS features** directly from WAV files using the Python API  
3. **Acoustic eGeMAPS features** from openSMILE CLI batch CSVs  

### Examples

```python
# 1) Linguistic + ICU/AOI from a transcript string
from speechmarc.features import build_text_feature_row

text = "The boy is reaching for the cookie jar while the sink is overflowing with water."
row = build_text_feature_row(text)  # pandas Series
print(row.head())

# 2) Acoustic eGeMAPS for a WAV file (Python API)
from speechmarc.features import extract_acoustic_egeMAPS
df_acoustic = extract_acoustic_egeMAPS("path/to/audio.wav")  # 1-row DataFrame

# 3) Acoustic eGeMAPS if you used openSMILE CLI (batch CSVs)
from speechmarc.features import combine_opensmile_cli_dir
smile_df = combine_opensmile_cli_dir("path/to/smile_output_dir")  # many rows
