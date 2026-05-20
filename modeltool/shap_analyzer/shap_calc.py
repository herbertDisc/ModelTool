from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


SHAP_POSITIVE_COLOR = "#4c78a8"
SHAP_NEGATIVE_COLOR = "#e45756"


def load_booster(model_path: str | Path) -> tuple[xgb.Booster, list[str]]:
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"model_path does not exist: {path}")

    booster = xgb.Booster()
    booster.load_model(str(path))
    features = booster.feature_names
    if not features:
        raise ValueError("Model does not expose feature names.")
    return booster, list(features)


def validate_model_features(df: pd.DataFrame, features: list[str]) -> None:
    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise ValueError(f"DataFrame is missing model features: {missing_features}")


def make_dmatrix(df: pd.DataFrame, features: list[str]) -> xgb.DMatrix:
    feature_df = df.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return xgb.DMatrix(feature_df, feature_names=features)


def predict_model_score(df: pd.DataFrame, booster: xgb.Booster, features: list[str]) -> pd.Series:
    pred = booster.predict(make_dmatrix(df, features))
    return pd.Series(pred, index=df.index, name="model_pred")


def build_shap_detail(
    target_df: pd.DataFrame,
    base_df: pd.DataFrame,
    booster: xgb.Booster,
    features: list[str],
) -> pd.DataFrame:
    shap_target = booster.predict(make_dmatrix(target_df, features), pred_contribs=True)
    shap_base = booster.predict(make_dmatrix(base_df, features), pred_contribs=True)

    shap_cols = features + ["BIAS"]
    mean_target = pd.Series(shap_target.mean(axis=0), index=shap_cols)
    mean_base = pd.Series(shap_base.mean(axis=0), index=shap_cols)

    detail = pd.DataFrame(
        {
            "mean_shap_target": mean_target,
            "mean_shap_base": mean_base,
        }
    )
    detail["target_minus_base"] = detail["mean_shap_target"] - detail["mean_shap_base"]
    detail["abs_target_minus_base"] = detail["target_minus_base"].abs()
    detail["direction"] = np.where(detail["target_minus_base"] >= 0, "positive", "negative")

    feat_mean_target = target_df.loc[:, features].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    feat_mean_base = base_df.loc[:, features].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    detail["feature_mean_target"] = feat_mean_target.reindex(detail.index)
    detail["feature_mean_base"] = feat_mean_base.reindex(detail.index)
    detail["feature_target_minus_base"] = detail["feature_mean_target"] - detail["feature_mean_base"]

    detail = detail.drop(index="BIAS", errors="ignore")
    detail.insert(0, "feature", detail.index)
    return detail.sort_values(
        ["abs_target_minus_base", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def save_shap_chart(shap_top: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ranked_df = shap_top.sort_values(
        ["abs_target_minus_base", "feature"], ascending=[False, True]
    ).copy()
    plot_df = ranked_df.iloc[::-1]
    signed_values = plot_df["target_minus_base"]
    colors = np.where(signed_values >= 0, SHAP_POSITIVE_COLOR, SHAP_NEGATIVE_COLOR)
    max_abs = signed_values.abs().max()
    label_offset = float(max_abs * 0.015) if pd.notna(max_abs) and max_abs > 0 else 0.001

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], signed_values, color=colors)
    for _, row in plot_df.iterrows():
        value = row["target_minus_base"]
        ax.text(
            value + (label_offset if value >= 0 else -label_offset),
            row["feature"],
            f"{value:+.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    if pd.notna(max_abs) and max_abs > 0:
        ax.set_xlim(-max_abs * 1.12, max_abs * 1.12)
    ax.set_title("Top SHAP diff ranked by abs | signed target_minus_base")
    ax.set_xlabel("signed mean SHAP(target) - mean SHAP(base)")
    ax.set_ylabel("feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output
