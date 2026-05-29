from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


GROUP_ORDER = ["base", "target"]
GROUP_COLORS = {
    "base": "#4c78a8",
    "target": "#e45756",
}


def build_bins(series: pd.Series, nbins: int) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or valid.nunique() <= 1:
        return [-np.inf, np.inf]

    _, bins = pd.qcut(valid, nbins, labels=False, retbins=True, duplicates="drop")
    bins = np.array(bins, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return list(bins)


def apply_bins(series: pd.Series, bins: list[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    grp = pd.cut(numeric, bins=bins, labels=False, include_lowest=True)
    return grp.fillna(-1).astype(int)


def format_bin_edge(value: float) -> str:
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "inf"
    return f"{value:.4g}"


def format_bin_range(grp: int, bins: list[float]) -> str:
    if grp == -1:
        return "Missing"
    left = format_bin_edge(bins[grp])
    right = format_bin_edge(bins[grp + 1])
    return f"B{grp + 1}: [{left}, {right}]"


def build_bivar_tables(
    target_df: pd.DataFrame,
    base_df: pd.DataFrame,
    features: list[str],
    label_col: str,
    nbins: int,
) -> dict[str, pd.DataFrame]:
    tables = {}
    for feature in features:
        bins = build_bins(base_df[feature], nbins)
        tables[feature] = build_bivar_compare_table(
            target_df=target_df,
            base_df=base_df,
            feature=feature,
            label_col=label_col,
            bins=bins,
        )
    return tables


def build_bivar_compare_table(
    target_df: pd.DataFrame,
    base_df: pd.DataFrame,
    feature: str,
    label_col: str,
    bins: list[float],
) -> pd.DataFrame:
    all_groups = []
    for group_name, group_df in {"base": base_df, "target": target_df}.items():
        work = pd.DataFrame(
            {
                "feature_value": pd.to_numeric(group_df[feature], errors="coerce"),
                "label": pd.to_numeric(group_df[label_col], errors="coerce"),
            },
            index=group_df.index,
        )
        work["grp"] = apply_bins(work["feature_value"], bins)
        grouped = (
            work.groupby("grp", observed=False)
            .agg(
                pop_n=("label", "size"),
                label_n=("label", "count"),
                feature_mean=("feature_value", "mean"),
                label_rate=("label", "mean"),
            )
            .reset_index()
        )
        total_n = int(grouped["pop_n"].sum())
        grouped["pop_pct"] = grouped["pop_n"] / total_n if total_n else np.nan
        grouped["bin_label"] = grouped["grp"].apply(lambda x: format_bin_range(int(x), bins))
        grouped["feature"] = feature
        grouped["compare_group"] = group_name
        all_groups.append(grouped)
    return pd.concat(all_groups, ignore_index=True)


def make_bivar_wide_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group_name in GROUP_ORDER:
        sub = feature_table[feature_table["compare_group"] == group_name].copy()
        sub = sub[["grp", "bin_label", "pop_n", "pop_pct", "label_n", "feature_mean", "label_rate"]]
        sub.columns = ["grp", "bin_label"] + [
            f"{group_name}_{col}"
            for col in ["pop_n", "pop_pct", "label_n", "feature_mean", "label_rate"]
        ]
        parts.append(sub)

    wide = parts[0]
    for sub in parts[1:]:
        wide = wide.merge(sub, on=["grp", "bin_label"], how="outer")
    return wide.sort_values("grp").reset_index(drop=True)


def save_bivar_chart(feature: str, feature_table: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_rate = plt.subplots(figsize=(12, 4.8))
    ax_pop = ax_rate.twinx()

    grp_values = sorted(feature_table["grp"].astype(int).unique())
    x = np.arange(len(grp_values))
    width = 0.28
    tick_labels = []
    for grp in grp_values:
        tick_labels.append("Missing" if grp == -1 else f"B{grp + 1}")

    for idx, group_name in enumerate(GROUP_ORDER):
        sub = feature_table[feature_table["compare_group"] == group_name].sort_values("grp")
        sub = sub.set_index("grp").reindex(grp_values).reset_index()
        offset = (idx - 0.5) * width
        ax_pop.bar(
            x + offset,
            sub["pop_pct"],
            width=width,
            alpha=0.24,
            color=GROUP_COLORS[group_name],
            label=f"{group_name} pop_pct",
        )
        ax_rate.plot(
            x + offset,
            sub["label_rate"],
            marker="o",
            linewidth=1.8,
            color=GROUP_COLORS[group_name],
            label=f"{group_name} label_rate",
        )

    ax_rate.set_title(f"{feature} | label rate and population share")
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(tick_labels, rotation=0)
    ax_rate.set_ylabel("label rate")
    ax_rate.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_pop.set_ylabel("population %")
    ax_pop.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_rate.grid(axis="y", alpha=0.25)

    handles_rate, labels_rate = ax_rate.get_legend_handles_labels()
    handles_pop, labels_pop = ax_pop.get_legend_handles_labels()
    ax_rate.legend(
        handles_rate + handles_pop,
        labels_rate + labels_pop,
        loc="upper center",
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def safe_image_name(feature: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("_") or "feature"
