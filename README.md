<div align="center">

SPEECH-MARC
Speech-based Multimodal Assessment of Risk for Cognition

An open-source pipeline for extracting linguistic and acoustic speech features, residualizing covariates, performing transparent feature selection, and evaluating baseline models for MoCA-defined MCI classification.

</div>

Table of Contents
Features

Install

Quickstart (Python API)

CLI Usage

Preprocessing & Feature Extraction

Residualization & Feature Selection

Evaluation

Method Notes

Module Map

License

Features
Preprocessing: transcript generation (Whisper) and demographics merge; audio duration via ffmpeg.

Feature Extraction

Linguistic: readability, POS rates, disfluency, deixis, ICU/AOI flags.

Acoustic: eGeMAPS (openSMILE Python API or CLI CSVs).

Residualization: regress out covariates (e.g., age, sex, education, duration) → _resid features.

Statistical Selection: Welch’s t, Cohen’s d, −log10(p), optional Pearson r to MoCA.

Evaluation: stratified 5-fold CV; L1-Logistic and linear SVM; AUC, accuracy, F1, sensitivity, specificity, precision.

Both a clean Python API and CLI for non-coders.

Install
bash
# Install the package directly from GitHub
pip install git+https://github.com/cynthianyongesa/SPEECH-MARC.git

# spaCy English model (for linguistic features)
python -m spacy download en_core_web_sm

# openSMILE (Python API) for acoustic features
pip install opensmile

# ffmpeg (system-level) for audio duration in preprocessing
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
The package also depends on: numpy, pandas, scikit-learn, matplotlib, scipy, tqdm, librosa, soundfile, sentence-transformers, torch, xgboost, catboost, shap, textstat, regex, click (All listed in pyproject.toml)

Quickstart (Python API)
python
import pandas as pd
from speechmarc.residualize import ResidualizeConfig, residualize
from speechmarc.feature_select import GroupingRule, select_features_statistical, pick_top_k
from speechmarc.evaluate import evaluate_cv

# 0) DataFrame `df` with features, covariates, binary label (0/1), and MoCA (if used)
covars = ["telephonescreen_age_v2", "cri_years_education", "audio_duration_sec", "sex_male"]
feat_cols = [c for c in df.columns if c.startswith("feat_")]

# 1) Residualize covariates out of features
cfg = ResidualizeConfig(features=feat_cols, covariates=covars, one_hot_covariates=True)
df_resid = residualize(df, cfg)

# 2) Statistical selection using MoCA threshold (>=26 = Normal)
rule = GroupingRule(moca_col="moca_total", moca_threshold=26)
resid_cols = [c for c in df_resid.columns if c.endswith("_resid")]
stats = select_features_statistical(
    df_resid, resid_cols, rule,
    compute_pearson_to_moca=True,
    moca_col_for_pearson="moca_total"
)
top = pick_top_k(stats, k=18, p_max=0.05, min_abs_d=0.5)

# 3) Evaluate top residualized features with 5-fold CV
X = df_resid[top["feature"].tolist()]
y = df_resid["label"].astype(int)
results = evaluate_cv(X, y)
print(results)
CLI Usage
1. Residualize features against covariates
bash
speechmarc-residualize \
  --in-csv data/features_with_covars.csv \
  --out-csv out/residualized.csv \
  --feature-cols '["feat_a","feat_b","feat_c"]' \
  --covariate-cols '["telephonescreen_age_v2","cri_years_education","audio_duration_sec","sex_male"]'
2. Statistical feature selection
bash
speechmarc-select \
  --in-csv out/residualized.csv \
  --out-stats-csv out/feature_stats.csv \
  --out-top-csv out/top_features.csv \
  --feature-cols '["feat_a_resid","feat_b_resid","feat_c_resid"]' \
  --moca-col moca_total --moca-threshold 26 --pearson --k 18 --p-max 0.05 --min-abs-d 0.5
3. Cross-validated evaluation
bash
speechmarc-eval \
  --in-csv out/residualized.csv \
  --label-col label \
  --feature-cols "$(python - <<'PY'
import pandas as pd, json
print(json.dumps(pd.read_csv('out/top_features.csv')['feature'].tolist()))
PY)"
Or replace with a static list:

bash
--feature-cols '["f1_resid","f2_resid","f3_resid"]'
Preprocessing & Feature Extraction
Preprocessing Utilities
Whisper transcription (per-file .txt)

Combined transcript CSV

Merge with demographics

Audio duration via ffmpeg

Linguistic + ICU/AOI from text
python
from speechmarc.features import build_text_feature_row
row = build_text_feature_row("The boy reaches for the cookie jar while the sink overflows.")
Acoustic eGeMAPS from WAV (Python API)
python
from speechmarc.features import extract_acoustic_egeMAPS
df_ac = extract_acoustic_egeMAPS("path/to/audio.wav")  # 1-row DataFrame
If using openSMILE CLI
python
from speechmarc.features import combine_opensmile_cli_dir
smile_df = combine_opensmile_cli_dir("path/to/smile_output_dir")
Install spaCy model once:

bash
python -m spacy download en_core_web_sm
Residualization & Feature Selection
python
from speechmarc.residualize import ResidualizeConfig, residualize
from speechmarc.feature_select import GroupingRule, select_features_statistical, pick_top_k

cfg = ResidualizeConfig(features=feat_cols, covariates=covars, one_hot_covariates=True)
df_resid = residualize(df, cfg)

rule = GroupingRule(moca_col="moca_total", moca_threshold=26)
resid_cols = [c for c in df_resid.columns if c.endswith("_resid")]
stats = select_features_statistical(
    df_resid, resid_cols, rule,
    compute_pearson_to_moca=True,
    moca_col_for_pearson="moca_total"
)
top = pick_top_k(stats, k=18, p_max=0.05, min_abs_d=0.5)
Evaluation
python
from speechmarc.evaluate import evaluate_cv, compare_feature_sets

# Panel on combined matrix
cv_results = evaluate_cv(X_combined, y)
print(cv_results)

# Compare feature sets with Logistic_L1
sets = {"linguistic": X_ling, "acoustic": X_acou, "combined": X_combined}
set_results = compare_feature_sets(sets, y, model_name="Logistic_L1")
print(set_results)
Method Notes
Combined linguistic + acoustic features outperform single-modality models.

L1-regularized Logistic Regression and linear-kernel SVM yield strongest AUCs (≈ 0.81).

Linguistic-only is moderate; acoustic-only is weaker. Fusion delivers complementary gains.

Module Map
Module	Description
speechmarc.preprocessing	transcript + demographics merge
speechmarc.features	linguistic / ICU-AOI / acoustic extraction
speechmarc.residualize	covariate adjustment
speechmarc.feature_select	statistical selection
speechmarc.evaluate	CV metrics, ROC utilities
speechmarc.cli	CLI commands
License
MIT © 2025 Cynthia Nyongesa
