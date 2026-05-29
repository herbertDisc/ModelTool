from __future__ import annotations

import numpy as np
import pandas as pd


PROFILE_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
FEATURE_PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def build_overview(
    df: pd.DataFrame,
    target_mask: pd.Series,
    base_mask: pd.Series,
    model_pred: pd.Series,
    score_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    group_masks = {
        "all": pd.Series(True, index=df.index),
        "target": target_mask,
        "base": base_mask,
    }
    for group_name, mask in group_masks.items():
        group_df = df.loc[mask]
        pred = pd.to_numeric(model_pred.loc[mask], errors="coerce")
        row = {
            "group_name": group_name,
            "total_count": int(mask.sum()),
            "share_of_all": float(mask.mean()),
            "model_pred_mean": pred.mean(),
            "model_pred_p10": pred.quantile(0.10),
            "model_pred_p50": pred.quantile(0.50),
            "model_pred_p90": pred.quantile(0.90),
        }
        if score_col is not None:
            score = pd.to_numeric(group_df[score_col], errors="coerce")
            row[f"{score_col}_mean"] = score.mean()
            row[f"{score_col}_p50"] = score.quantile(0.50)
        if label_col is not None and label_col in group_df.columns:
            label = pd.to_numeric(group_df[label_col], errors="coerce")
            row[f"{label_col}_mean"] = label.mean()
            row[f"{label_col}_n"] = int(label.notna().sum())
        if "order_dt" in group_df.columns:
            row["order_dt_min"] = group_df["order_dt"].min()
            row["order_dt_max"] = group_df["order_dt"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def build_score_profile(
    df: pd.DataFrame,
    target_mask: pd.Series,
    base_mask: pd.Series,
    model_pred: pd.Series,
    score_col: str | None = None,
) -> pd.DataFrame:
    score_sources = {"model_pred": model_pred}
    if score_col is not None:
        score_sources[score_col] = df[score_col]

    rows = []
    group_masks = {
        "all": pd.Series(True, index=df.index),
        "target": target_mask,
        "base": base_mask,
    }
    for score_name, values in score_sources.items():
        numeric_values = pd.to_numeric(values, errors="coerce")
        for group_name, mask in group_masks.items():
            series = numeric_values.loc[mask].dropna()
            row = {
                "score_name": score_name,
                "group_name": group_name,
                "count": int(mask.sum()),
                "n": int(series.count()),
                "missing_rate": 1 - float(series.count() / mask.sum()) if mask.sum() else np.nan,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
            }
            for percentile in PROFILE_PERCENTILES:
                row[f"p{int(percentile * 100):02d}"] = series.quantile(percentile)
            rows.append(row)
    return pd.DataFrame(rows)


def means_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    feature_df = df.loc[:, features].apply(pd.to_numeric, errors="coerce")
    res = pd.DataFrame(index=features)
    res["total_count"] = len(df)
    res["n"] = feature_df.notna().sum(axis=0)
    res["missing_rate"] = feature_df.isna().mean(axis=0)
    res["mean"] = feature_df.mean(axis=0)
    for percentile in FEATURE_PERCENTILES:
        res[f"p{int(percentile * 100)}"] = feature_df.quantile(percentile)
    return res


def build_feature_profile(
    target_df: pd.DataFrame,
    base_df: pd.DataFrame,
    features: list[str],
    ordered_features: list[str],
) -> pd.DataFrame:
    ordered = [feature for feature in ordered_features if feature in features]
    target_profile = means_report(target_df, ordered)
    base_profile = means_report(base_df, ordered)

    target_profile.columns = [f"target_{col}" for col in target_profile.columns]
    base_profile.columns = [f"base_{col}" for col in base_profile.columns]
    merged = pd.concat([target_profile, base_profile], axis=1)
    merged["target_minus_base_mean"] = merged["target_mean"] - merged["base_mean"]
    merged["target_minus_base_missing_rate"] = (
        merged["target_missing_rate"] - merged["base_missing_rate"]
    )
    merged.insert(0, "feature", merged.index)
    return merged.reset_index(drop=True)
