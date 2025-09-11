"""
Command-line interface for SPEECH-MARC

Commands:
- speechmarc-residualize : residualize feature columns against covariates
- speechmarc-select      : statistical feature selection (Welch t, Cohen's d, -log10(p), Pearson r)
- speechmarc-eval        : cross-validated evaluation (AUC, accuracy, F1, sensitivity, specificity, precision)
"""

from __future__ import annotations
import json
from pathlib import Path
import click
import pandas as pd

from .residualize import ResidualizeConfig, residualize
from .feature_select import GroupingRule, select_features_statistical, pick_top_k
from .evaluate import evaluate_cv, model_presets


@click.group()
def cli():
    """SPEECH-MARC command-line tools."""
    pass


# ----------------------------- Residualize ----------------------------------

@cli.command("residualize")
@click.option("--in-csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV with features + covariates.")
@click.option("--out-csv", type=click.Path(dir_okay=False), required=True, help="Output CSV with appended *_resid columns.")
@click.option("--feature-cols", required=True, help="JSON list or comma-separated feature column names.")
@click.option("--covariate-cols", required=True, help="JSON list or comma-separated covariate column names.")
@click.option("--no-intercept", is_flag=True, help="Disable intercept term in residualization.")
@click.option("--no-onehot", is_flag=True, help="Disable one-hot encoding for categorical covariates.")
@click.option("--suffix", default="_resid", show_default=True, help="Suffix for residualized columns.")
def residualize_cmd(in_csv, out_csv, feature_cols, covariate_cols, no_intercept, no_onehot, suffix):
    df = pd.read_csv(in_csv)

    def parse_cols(s: str):
        s = s.strip()
        if s.startswith("["):
            return json.loads(s)
        return [c.strip() for c in s.split(",") if c.strip()]

    feats = parse_cols(feature_cols)
    covs  = parse_cols(covariate_cols)

    cfg = ResidualizeConfig(
        features=feats,
        covariates=covs,
        add_intercept=not no_intercept,
        one_hot_covariates=not no_onehot,
        suffix=suffix,
    )
    out = residualize(df, cfg)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    click.echo(f"Saved residualized CSV → {out_csv}")


# ----------------------------- Select --------------------------------------

@cli.command("select")
@click.option("--in-csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV (e.g., residualized features).")
@click.option("--out-stats-csv", type=click.Path(dir_okay=False), required=True, help="Per-feature statistics output CSV.")
@click.option("--out-top-csv", type=click.Path(dir_okay=False), required=False, help="Optional: save top-k selection CSV.")
@click.option("--feature-cols", required=True, help="JSON list or comma-separated feature column names to test.")
@click.option("--label-col", default=None, help="Binary label column (0/1). If omitted, use MoCA threshold.")
@click.option("--positive-label", default=None, help="Value considered 'positive'/Normal in label_col.")
@click.option("--moca-col", default=None, help="MoCA column name (for thresholding).")
@click.option("--moca-threshold", type=float, default=None, help="Threshold for Normal vs MCI (e.g., 26).")
@click.option("--k", type=int, default=18, show_default=True, help="Top-k to keep (if --out-top-csv provided).")
@click.option("--p-max", type=float, default=0.05, show_default=True, help="Max p-value for top-k filter.")
@click.option("--min-abs-d", type=float, default=0.0, show_default=True, help="Min |Cohen's d| for top-k filter.")
@click.option("--pearson", is_flag=True, help="Also compute Pearson r to MoCA.")
@click.option("--pearson-col", default="moca_total", show_default=True, help="MoCA column for Pearson r.")
def select_cmd(in_csv, out_stats_csv, out_top_csv, feature_cols, label_col, positive_label, moca_col, moca_threshold, k, p_max, min_abs_d, pearson, pearson_col):
    df = pd.read_csv(in_csv)

    def parse_cols(s: str):
        s = s.strip()
        if s.startswith("["):
            return json.loads(s)
        return [c.strip() for c in s.split(",") if c.strip()]

    feats = parse_cols(feature_cols)

    rule = GroupingRule(
        label_col=label_col if label_col else None,
        positive_label=positive_label,
        moca_col=moca_col if moca_col else None,
        moca_threshold=moca_threshold,
    )
    stats = select_features_statistical(
        df, feats, rule,
        compute_pearson_to_moca=bool(pearson),
        moca_col_for_pearson=pearson_col,
    )
    Path(out_stats_csv).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_stats_csv, index=False)
    click.echo(f"Saved stats CSV → {out_stats_csv}")

    if out_top_csv:
        top = pick_top_k(stats, k=k, p_max=p_max, min_abs_d=min_abs_d)
        top.to_csv(out_top_csv, index=False)
        click.echo(f"Saved top-k CSV → {out_top_csv}")


# ----------------------------- Evaluate ------------------------------------

@cli.command("eval")
@click.option("--in-csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV with features + label column.")
@click.option("--label-col", required=True, help="Binary label column (0/1).")
@click.option("--feature-cols", required=True, help="JSON list or comma-separated feature columns to use.")
@click.option("--n-splits", type=int, default=5, show_default=True, help="Stratified CV folds.")
@click.option("--out-results-csv", type=click.Path(dir_okay=False), required=False, help="Optional: save metrics table.")
def eval_cmd(in_csv, label_col, feature_cols, n_splits, out_results_csv):
    df = pd.read_csv(in_csv)

    def parse_cols(s: str):
        s = s.strip()
        if s.startswith("["):
            return json.loads(s)
        return [c.strip() for c in s.split(",") if c.strip()]

    feats = parse_cols(feature_cols)
    X = df[feats].copy()
    y = df[label_col].astype(int)

    results = evaluate_cv(X, y, models=model_presets(), n_splits=n_splits)
    click.echo(results.to_string(index=False))

    if out_results_csv:
        Path(out_results_csv).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_results_csv, index=False)
        click.echo(f"Saved results CSV → {out_results_csv}")


if __name__ == "__main__":
    cli()
