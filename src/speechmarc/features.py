"""
Feature extraction for SPEECH-MARC.

Includes:
- Acoustic features from openSMILE (Python API), plus a loader for openSMILE CLI CSVs
- Linguistic features (readability, TTR, POS rates, negation, hedging, disfluency, deixis)
- ICU/AOI features for the modern Cookie Theft scene

Notes:
- For acoustic features with the Python API, you need `opensmile`.
- For linguistic features, you need `spacy` (and a model, e.g. en_core_web_sm),
  plus `textstat` and `nltk` (with punkt tokenizer).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import re

# ---------- Optional imports (soft-fail so the rest of the package still imports) ----------
try:
    import opensmile  # Python API for openSMILE
except Exception:  # pragma: no cover
    opensmile = None  # type: ignore

try:
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
    Language = None  # type: ignore

try:
    import nltk
    from nltk import FreqDist
except Exception:  # pragma: no cover
    nltk = None  # type: ignore
    FreqDist = None  # type: ignore

try:
    import textstat
except Exception:  # pragma: no cover
    textstat = None  # type: ignore


# ==========================================================================================
# Acoustic features
# ==========================================================================================

def extract_acoustic_egeMAPS(wav_path: str | Path) -> pd.DataFrame:
    """
    Extract eGeMAPS Functionals via the openSMILE Python API for ONE wav file.
    Returns a single-row DataFrame indexed by file path.

    If `opensmile` isn't installed, raises ImportError.
    """
    if opensmile is None:
        raise ImportError("opensmile is not installed. `pip install opensmile`")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(str(wav_path))
    df.index = [Path(wav_path).name]
    return df


def load_opensmile_cli_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV produced by the openSMILE CLI (emobase/eGeMAPS configs).
    Many of these CSVs contain ARFF-style headers with an '@data' line;
    this parses column names and the numeric data region robustly.

    Returns a 1-row DataFrame with a 'filename' column.
    """
    lines = Path(csv_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # find @data start
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = i + 1
            break
    if data_start is None:
        # Fall back to naive CSV read
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            df["filename"] = Path(csv_path).name
        return df

    # column names from '@attribute name type' lines
    attr_lines = [ln for ln in lines if ln.strip().lower().startswith("@attribute ")]
    col_names = []
    for ln in attr_lines:
        parts = ln.split()
        # @attribute <name> <type>
        if len(parts) >= 3:
            col_names.append(parts[1])

    df = pd.read_csv(csv_path, sep=",", header=None, skiprows=data_start)
    if len(col_names) >= df.shape[1]:
        df.columns = col_names[: df.shape[1]]
    df["filename"] = Path(csv_path).name
    return df


def combine_opensmile_cli_dir(output_dir: str | Path) -> pd.DataFrame:
    """
    Combine all openSMILE CLI CSVs in a directory into one DataFrame.
    Assumes each CSV corresponds to one file (1 row with many features).
    """
    output_dir = Path(output_dir)
    dfs = []
    for p in sorted(output_dir.glob("*.csv")):
        try:
            dfs.append(load_opensmile_cli_csv(p))
        except Exception as e:  # pragma: no cover
            print(f"Skipping {p.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ==========================================================================================
# Linguistic features (hybrid of your notebook metrics)
# ==========================================================================================

# Minimal, curated term lists (extend as needed)
_DISFLUENCIES = [
    "um", "uh", "er", "ah", "hmm", "uhh", "umm", "like", "you know", "well", "i mean"
]
_TEMPORAL_DEIXIS = [
    "yesterday", "today", "now", "currently", "this morning", "tomorrow", "later"
]
_SPATIAL_DEIXIS = [
    "here", "there", "this", "that", "these", "those", "left", "right", "nearby"
]
_HEDGES = [
    "probably", "possibly", "seems", "likely", "may", "might", "apparently", "perhaps", "could", "sort of", "kind of"
]

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def _ensure_spacy(nlp: Optional["Language"]) -> "Language":
    """Load en_core_web_sm if no spaCy pipeline is provided."""
    if nlp is not None:
        return nlp
    if spacy is None:
        raise ImportError("spaCy is not installed. `pip install spacy` and download a model.")
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm") from e


def _readability(text: str) -> Dict[str, float]:
    if textstat is None:
        return {}
    return {
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
        "gunning_fog_index": float(textstat.gunning_fog(text)),
        "coleman_liau_index": float(textstat.coleman_liau_index(text)),
        "dale_chall_score": float(textstat.dale_chall_readability_score(text)),
        "automated_readability_index": float(textstat.automated_readability_index(text)),
    }


def extract_linguistic_features(text: str, nlp: Optional["Language"] = None) -> Dict[str, float]:
    """
    Compute a compact set of linguistics features from a transcript string:
      - word_count, type_token_ratio, lexical_diversity (unique lemmas / tokens)
      - POS rates (nouns/verbs/adjectives/adverbs/pronouns per word)
      - negation count, passive auxiliaries, modality (MD)
      - disfluency / hedge / deixis counts
      - readability indices (if textstat is installed)

    Mirrors/condenses your NLP notebook metrics into a stable feature set.
    """
    text = str(text or "")
    out: Dict[str, float] = {}
    out["char_count"] = float(len(text))

    # quick lower once
    low = text.lower()

    # disfluencies / hedge / deixis
    out["disfluency_count"] = float(sum(low.count(tok) for tok in _DISFLUENCIES))
    out["hedge_count"] = float(sum(low.count(tok) for tok in _HEDGES))
    out["temporal_deixis_count"] = float(sum(low.count(tok) for tok in _TEMPORAL_DEIXIS))
    out["spatial_deixis_count"] = float(sum(low.count(tok) for tok in _SPATIAL_DEIXIS))

    # spaCy-driven metrics
    nlp = _ensure_spacy(nlp)
    doc = nlp(text)

    tokens_alpha = [t for t in doc if t.is_alpha]
    n_words = len(tokens_alpha)
    out["word_count"] = float(n_words)

    if n_words > 0:
        lemmas = {t.lemma_ for t in tokens_alpha}
        out["type_token_ratio"] = float(len(set(t.text for t in tokens_alpha)) / n_words)
        out["lexical_diversity"] = float(len(lemmas) / n_words)
    else:
        out["type_token_ratio"] = 0.0
        out["lexical_diversity"] = 0.0

    # POS rates
    def rate(pos_name: str) -> float:
        if n_words == 0:
            return 0.0
        return sum(1 for t in tokens_alpha if t.pos_ == pos_name) / n_words

    out["nouns_rate"] = rate("NOUN")
    out["verbs_rate"] = rate("VERB")
    out["adjectives_rate"] = rate("ADJ")
    out["adverbs_rate"] = rate("ADV")
    out["pronouns_rate"] = rate("PRON")

    # modality / negation / passive aux
    out["modality_md_count"] = float(sum(1 for t in doc if t.tag_ == "MD"))
    out["negation_count"] = float(sum(1 for t in doc if t.dep_ == "neg"))
    out["auxpass_count"] = float(sum(1 for t in doc if t.dep_ == "auxpass"))

    # readability (if available)
    out.update(_readability(text))
    return out


# ==========================================================================================
# ICU / AOI features for the Cookie Theft scene
# ==========================================================================================

# Default ICU and AOI keyword sets (trimmed versions; expand as needed)
ICU_KEYWORDS: Dict[str, List[str]] = {
    "ICU_BoyOnStool": ["boy", "kid", "child", "stool", "step"],
    "ICU_CookieJar": ["cookie", "cookies", "jar", "reach", "grab"],
    "ICU_GirlEatingCookie": ["girl", "eating", "chew", "bite", "smile"],
    "ICU_DogLickingFloor": ["dog", "lick", "floor", "crumbs"],
    "ICU_ManDoingDishes": ["man", "dad", "father", "dishes", "wash", "sink"],
    "ICU_SinkOverflow": ["sink", "overflow", "water", "soap", "bubbles"],
    "ICU_WomanMowing": ["woman", "mom", "mow", "lawn", "grass", "phone"],
    "ICU_CatBirds": ["cat", "birds", "bird", "watch"],
    "ICU_YardScene": ["window", "yard", "outside", "fence", "building"],
    "ICU_Curtains": ["curtain", "pink", "pattern"],
    "ICU_SpilledWaterFloor": ["water", "spill", "wet", "puddle"],
    "ICU_KitchenDetails": ["cabinet", "drawer", "kitchen", "counter"],
    "ICU_ObjectsWithColor": ["red", "blue", "green", "pink", "yellow", "turquoise"],
    "ICU_RaceEthnicity": ["black", "white", "skin", "dark", "light"],
    "ICU_GenderNormCommentary": ["gender", "role", "mom", "dad", "normal"],
    "ICU_DetailRichness": ["striped", "sponge", "buckle", "sleeve"],
    "ICU_CharacterEmotion": ["smile", "happy", "laugh", "scared", "worried"],
    "ICU_MessChaos": ["mess", "chaos", "clutter", "disaster"],
    "ICU_ObjectActionPairs": ["grabbing", "licking", "eating", "washing"],
    "ICU_BlendNarrativeMeta": ["scene", "story", "roles", "family", "switched"],
}

AOI_KEYWORDS: Dict[str, List[str]] = {
    "AOI_KidsLeft": ["boy", "girl", "cookie", "jar", "stool", "dog", "crumbs"],
    "AOI_KitchenCenter": ["man", "dad", "sink", "plate", "dishes", "soap"],
    "AOI_BackyardRight": ["woman", "mom", "outside", "lawnmower", "phone", "cat", "birds"],
    "AOI_WindowArea": ["window", "building", "sky", "house", "outside"],
    "AOI_InteriorDetails": ["curtain", "cabinet", "drawer", "white"],
    "AOI_IdentityFocus": ["black", "white", "skin", "gender", "mom", "dad"],
    "AOI_ColorFocus": ["red", "blue", "green", "pink", "yellow", "turquoise"],
}


def extract_icu_aoi_features(
    text: str,
    icu_keywords: Dict[str, List[str]] = ICU_KEYWORDS,
    aoi_keywords: Dict[str, List[str]] = AOI_KEYWORDS,
) -> Dict[str, float]:
    """
    Map a transcript onto ICU (interpretive content units) and AOI (areas of interest)
    indicators + simple coverage metrics.
    """
    t = str(text or "").lower()

    icu_flags: Dict[str, int] = {}
    icu_freqs: Dict[str, int] = {}
    icu_count = 0

    for icu, kws in icu_keywords.items():
        freqs = [t.count(kw) for kw in kws]
        total = int(sum(freqs))
        icu_freqs[f"{icu}_Frequency"] = total
        found = int(total > 0)
        icu_flags[icu] = found
        if found:
            icu_count += 1

    aoi_flags: Dict[str, int] = {}
    for aoi, kws in aoi_keywords.items():
        aoi_flags[aoi] = int(any(kw in t for kw in kws))

    total_icu = max(1, len(icu_keywords))
    icu_coverage = icu_count / total_icu

    return {
        **icu_flags,
        **icu_freqs,
        **aoi_flags,
        "ICU_Count": float(icu_count),
        "ICU_Coverage": float(icu_coverage),
    }


# ==========================================================================================
# Combining linguistic + ICU/AOI for a single text
# ==========================================================================================

def build_text_feature_row(text: str, nlp: Optional["Language"] = None) -> pd.Series:
    """
    Build a single combined feature row (linguistic + ICU/AOI) from one transcript string.
    """
    ling = extract_linguistic_features(text, nlp=nlp)
    icu = extract_icu_aoi_features(text)
    return pd.Series({**ling, **icu})
