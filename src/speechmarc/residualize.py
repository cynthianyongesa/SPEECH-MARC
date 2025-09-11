"""
Residualization for SPEECH-MARC

Goal:
- Take a feature matrix and remove variance explained by covariates
  (e.g., age, sex, education, audio duration).

Notes:
- Works with numeric and categorical covariates (one-hot).
- Adds new columns with a `_resid` suffix rather than overwriting.
- Builds the design matrix once and concatenates once to avoid pandas
  fragmentation warnings.

Public API:
- residualize(df, cfg)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ResidualizeConfig:
    features: Sequence[str]
    covariates: Sequence[str]
    add_intercept: bool = True
    suffix: str = "_resid"
    one_hot_covariates: bool = True
    drop_first: bool = True  # for one-hot encoding


def _design_matrix(
    df: pd.DataFrame,
    cols: Sequence[str],
    one_hot: bool,
    drop_first: bool
) -> pd.DataFrame:
    X = df.loc[:, cols]
    if one_hot:
        X = pd.get_dummies(X, drop_first=drop_first)
    return X.apply(pd.to_numeric, errors="coerce")


def residualize(df: pd.DataFrame, cfg: ResidualizeConfig) -> pd.DataFrame:
    """
    For each feature y in cfg.features, fit y ~ covariates and return residuals (y - ŷ).
    Returns a copy of df with extra columns: <feature> + cfg.suffix.

    Example
    -------
    cfg = ResidualizeConfig(
        features=[c for c in df.columns if c.startswith("feat_")],
        covariates=["age", "sex_male", "edu_years", "audio_duration_sec"]
    )
    out = residualize(df, cfg)
    """
    out = df.copy()

    # Build covariate design once
    X = _design_matrix(df, cfg.covariates, cfg.one_hot_covariates, cfg.drop_first)
    if cfg.add_intercept:
        X = X.assign(_intercept=1.0)

    # OLS without extra intercept (we already appended it if requested)
    lr = LinearRegression(fit_intercept=False)
    X_values = X.values

    # Collect residuals into a single block to avoid fragmentation
    resid_blocks: dict[str, np.ndarray] = {}

    for feat in cfg.features:
        y = pd.to_numeric(df[feat], errors="coerce").values.reshape(-1, 1)
        mask = np.isfinite(y).ravel() & np.isfinite(X_values).all(axis=1)

        if mask.sum() < 3:
            resid = np.full_like(y, np.nan, dtype=float).ravel()
        else:
            lr.fit(X_values[mask, :], y[mask])
            yhat = lr.predict(X_values)
            resid = (y - yhat).ravel()

        resid_blocks[f"{feat}{cfg.suffix}"] = resid

    resid_df = pd.DataFrame(resid_blocks, index=df.index)
    return pd.concat([out, resid_df], axis=1)
