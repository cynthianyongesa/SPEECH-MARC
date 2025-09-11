"""
Feature selection for SPEECH-MARC

What this does
--------------
Given a table of features and labels (or MoCA scores), compute simple,
transparent statistics to rank features:
- Welch's t-test (robust to unequal variances)
- Cohen's d (effect size)
- -log10(p) (useful for quick ranking/plots)
- Optional Pearson r to MoCA (continuous)

Outputs a tidy DataFrame with consistent columns so it slots into plots/tables.

Public API
----------
- select_features_statistical(df, feature_cols, rule, ...)
- pick_top_k(stats, k=..., p_max=..., min_abs_d=...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr


# ----- How we split groups for the t-test ------------------------------------

@dataclass
class GroupingRule:
    """
    Define how to split rows into two groups.

    Option A: use an existing binary label column (e.g., 'label' 0/1)
        label_col='label', positive_label=1  # 1 = "Normal" (or whichever you prefer)

    Option B: derive groups from MoCA threshold (e.g., >=26 Normal, else MCI)
        moca_col='moca_total', moca_threshold=26
    """
    # Option A: existing label column
    label_col: Optional[str] = None
    positive_label: Optional[object] = None   # which value counts as the "positive"/Normal group

    # Option B: threshold on a continuous MoCA score
    moca_col: Optional[str] = None
    moca_threshold: Optional[float] = None


def _split_groups(df: pd.DataFrame, rule: GroupingRule) -> Tuple[pd.Series, pd.Series]:
    """Return boolean masks (g_pos, g_neg) selecting the two groups."""
    if rule.label_col:
        lbl = df[rule.label_col]
        pos = rule.positive_label
        if pos is None:
            # default: pick the majority class as "positive"
            counts = lbl.value_counts()
            if len(counts) < 2:
                raise ValueError("Label column must contain two classes.")
            pos = counts.idxmax()
        return (lbl == pos), (lbl != pos)

    if rule.moca_col and rule.moca_threshold is not None:
        moca = pd.to_numeric(df[rule.moca_col], errors="coerce")
        return (moca >= rule.moca_threshold), (moca < rule.moca_threshold)

    raise ValueError("Provide either label_col or (moca_col + moca_threshold) in GroupingRule.")


# ----- Stats helpers ---------------------------------------------------------

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d; safe on small/constant inputs."""
    x = x.astype(float)
    y = y.astype(float)
    sx2 = np.nanvar(x, ddof=1)
    sy2 = np.nanvar(y, ddof=1)
    pooled = np.sqrt((sx2 + sy2) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return (np.nanmean(x) - np.nanmean(y)) / pooled


# ----- Main entry point ------------------------------------------------------

def select_features_statistical(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    rule: GroupingRule,
    compute_pearson_to_moca: bool = True,
    moca_col_for_pearson: str = "moca_total",
) -> pd.DataFrame:
    """
    Compute stats per feature and return a tidy, sorted DataFrame with:
    ['feature', 't_stat', 'p_value', 'cohens_d', 'neg_log10_p', 'pearson_r'(optional)]

    Sorted by ascending p_value (NaNs go to the bottom).
    """
    g_pos, g_neg = _split_groups(df, rule)

    # Optional MoCA vector for correlations
    moca = None
    if compute_pearson_to_moca:
        moca = pd.to_numeric(df[moca_col_for_pearson], errors="coerce")

    rows = []
    for col in feature_cols:
        x = pd.to_numeric(df.loc[g_pos, col], errors="coerce").dropna().values
        y = pd.to_numeric(df.loc[g_neg, col], errors="coerce").dropna().values

        # Welch's t-test (equal_var=False)
        if len(x) > 1 and len(y) > 1:
            t_stat, p = ttest_ind(x, y, equal_var=False)
        else:
            t_stat, p = np.nan, np.nan

        d = _cohens_d(x if len(x) else np.array([np.nan]),
                      y if len(y) else np.array([np.nan]))

        r = np.nan
        if moca is not None:
            pair = pd.DataFrame({"moca": moca, "feat": pd.to_numeric(df[col], errors="coerce")}).dropna()
            if len(pair) > 2 and pair["feat"].nunique() > 1:
                try:
                    r, _ = pearsonr(pair["moca"], pair["feat"])
                except Exception:
                    r = np.nan

        rows.append({
            "feature": col,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
            "cohens_d": float(d) if np.isfinite(d) else np.nan,
            "neg_log10_p": float(-np.log10(p)) if p and p > 0 else np.nan,
            "pearson_r": float(r) if np.isfinite(r) else np.nan,
        })

    stats = pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)
    return stats


def pick_top_k(stats: pd.DataFrame, k: int = 18, p_max: float = 0.05, min_abs_d: float = 0.0) -> pd.DataFrame:
    """
    Filter by thresholds and return the top-k rows.
    - p_max: keep features with p <= p_max
    - min_abs_d: keep features with |d| >= min_abs_d
    """
    f = stats.copy()
    if "p_value" in f:
        f = f[f["p_value"].notna() & (f["p_value"] <= p_max)]
    if "cohens_d" in f:
        f = f[f["cohens_d"].abs() >= min_abs_d]
    return f.head(k).reset_index(drop=True)
