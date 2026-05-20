from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .bivar import build_bivar_tables, safe_image_name, save_bivar_chart
from .excel import export_excel
from .profiles import build_feature_profile, build_overview, build_score_profile
from .selectors import Selector, build_population_split, resolve_selector, selector_description
from .shap_calc import (
    build_shap_detail,
    load_booster,
    predict_model_score,
    save_shap_chart,
    validate_model_features,
)


@dataclass
class ShapDriftResult:
    overview: pd.DataFrame
    score_profile: pd.DataFrame
    shap_detail: pd.DataFrame
    shap_top: pd.DataFrame
    feature_profile: pd.DataFrame
    bivar_tables: dict[str, pd.DataFrame]
    config: dict[str, Any]
    output_xlsx: Path
    shap_chart_path: Path
    bivar_image_paths: dict[str, Path]


def build_shap_drift_report(
    *,
    df: pd.DataFrame,
    model_path: str | Path,
    target: Selector,
    base: Selector,
    output_xlsx: str | Path | None = None,
    label_col: str | None = None,
    mature: Selector | None = None,
    score_col: str | None = None,
    top_n: int = 20,
    bivar_nbins: int = 5,
    image_dir: str | Path | None = None,
) -> ShapDriftResult:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df cannot be empty.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if bivar_nbins <= 1:
        raise ValueError("bivar_nbins must be greater than 1.")

    output_path = _resolve_output_path(output_xlsx)
    image_path = _resolve_image_dir(output_path, image_dir)

    split = build_population_split(df, target, base)
    booster, features = load_booster(model_path)
    validate_model_features(df, features)
    if score_col is not None and score_col not in df.columns:
        raise ValueError(f"score_col does not exist in DataFrame: {score_col}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"label_col does not exist in DataFrame: {label_col}")

    model_pred = predict_model_score(df, booster, features)
    overview = build_overview(
        df=df,
        target_mask=split.target_mask,
        base_mask=split.base_mask,
        model_pred=model_pred,
        score_col=score_col,
        label_col=label_col,
    )
    score_profile = build_score_profile(
        df=df,
        target_mask=split.target_mask,
        base_mask=split.base_mask,
        model_pred=model_pred,
        score_col=score_col,
    )
    shap_detail = build_shap_detail(
        target_df=split.target_df,
        base_df=split.base_df,
        booster=booster,
        features=features,
    )
    shap_top = shap_detail.head(top_n).copy()
    feature_profile = build_feature_profile(
        target_df=split.target_df,
        base_df=split.base_df,
        features=features,
        ordered_features=shap_detail["feature"].tolist(),
    )

    image_path.mkdir(parents=True, exist_ok=True)
    shap_chart_path = save_shap_chart(shap_top, image_path / "top_shap_target_minus_base.png")

    bivar_tables: dict[str, pd.DataFrame] = {}
    bivar_image_paths: dict[str, Path] = {}
    mature_mask = None
    if label_col is not None:
        if mature is not None:
            mature_mask = resolve_selector(df, mature, "mature")
        else:
            mature_mask = pd.Series(True, index=df.index)
        target_bivar_mask = split.target_mask & mature_mask
        base_bivar_mask = split.base_mask & mature_mask
        if not target_bivar_mask.any():
            raise ValueError("target sample is empty after applying mature selector for bivar.")
        if not base_bivar_mask.any():
            raise ValueError("base sample is empty after applying mature selector for bivar.")

        bivar_target_df = df.loc[target_bivar_mask].copy()
        bivar_base_df = df.loc[base_bivar_mask].copy()
        bivar_features = shap_detail["feature"].tolist()
        bivar_tables = build_bivar_tables(
            target_df=bivar_target_df,
            base_df=bivar_base_df,
            features=bivar_features,
            label_col=label_col,
            nbins=bivar_nbins,
        )
        for feature, table in bivar_tables.items():
            output_image = image_path / f"{safe_image_name(feature)}_bivar.png"
            bivar_image_paths[feature] = save_bivar_chart(feature, table, output_image)

    config = {
        "generated_at": datetime.now(),
        "model_path": str(Path(model_path)),
        "output_xlsx": str(output_path),
        "image_dir": str(image_path),
        "target_selector": selector_description(target),
        "base_selector": selector_description(base),
        "mature_selector": selector_description(mature),
        "target_count": int(split.target_mask.sum()),
        "base_count": int(split.base_mask.sum()),
        "all_count": len(df),
        "score_col": score_col,
        "label_col": label_col,
        "top_n": top_n,
        "bivar_nbins": bivar_nbins,
        "bivar_enabled": label_col is not None,
        "bivar_feature_count": len(bivar_tables),
        "bivar_target_count": int((split.target_mask & mature_mask).sum()) if mature_mask is not None else 0,
        "bivar_base_count": int((split.base_mask & mature_mask).sum()) if mature_mask is not None else 0,
        "model_feature_count": len(features),
    }

    export_excel(
        overview=overview,
        score_profile=score_profile,
        shap_detail=shap_detail,
        shap_top=shap_top,
        feature_profile=feature_profile,
        bivar_tables=bivar_tables,
        config=config,
        output_xlsx=output_path,
        shap_chart_path=shap_chart_path,
        bivar_image_paths=bivar_image_paths,
    )

    return ShapDriftResult(
        overview=overview,
        score_profile=score_profile,
        shap_detail=shap_detail,
        shap_top=shap_top,
        feature_profile=feature_profile,
        bivar_tables=bivar_tables,
        config=config,
        output_xlsx=output_path,
        shap_chart_path=shap_chart_path,
        bivar_image_paths=bivar_image_paths,
    )


def _resolve_output_path(output_xlsx: str | Path | None) -> Path:
    if output_xlsx is not None:
        return Path(output_xlsx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"shap_drift_report_{timestamp}.xlsx"


def _resolve_image_dir(output_xlsx: Path, image_dir: str | Path | None) -> Path:
    if image_dir is not None:
        return Path(image_dir)
    return output_xlsx.parent / f"{output_xlsx.stem}_images"
