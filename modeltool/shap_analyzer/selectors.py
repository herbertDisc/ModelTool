from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


Selector = str | pd.Series | np.ndarray | Sequence[bool]


@dataclass(frozen=True)
class PopulationSplit:
    target_mask: pd.Series
    base_mask: pd.Series
    target_df: pd.DataFrame
    base_df: pd.DataFrame


def selector_description(selector: Selector | None) -> str | None:
    if selector is None:
        return None
    if isinstance(selector, str):
        return selector
    return f"<{type(selector).__name__} boolean mask>"


def resolve_selector(df: pd.DataFrame, selector: Selector, name: str) -> pd.Series:
    if selector is None:
        raise ValueError(f"{name} selector is required.")

    if isinstance(selector, str):
        expression = selector.strip()
        if not expression:
            raise ValueError(f"{name} selector cannot be empty.")
        try:
            raw_mask = df.eval(expression, engine="python")
        except Exception as exc:
            raise ValueError(f"{name} selector failed to evaluate: {expression}") from exc
    else:
        raw_mask = selector

    return _coerce_boolean_mask(df, raw_mask, name)


def build_population_split(df: pd.DataFrame, target: Selector, base: Selector) -> PopulationSplit:
    target_mask = resolve_selector(df, target, "target")
    base_mask = resolve_selector(df, base, "base")

    target_count = int(target_mask.sum())
    base_count = int(base_mask.sum())
    if target_count == 0:
        raise ValueError("target selector produced an empty sample.")
    if base_count == 0:
        raise ValueError("base selector produced an empty sample.")

    overlap_count = int((target_mask & base_mask).sum())
    if overlap_count:
        raise ValueError(f"target and base selectors overlap on {overlap_count} rows.")

    return PopulationSplit(
        target_mask=target_mask,
        base_mask=base_mask,
        target_df=df.loc[target_mask].copy(),
        base_df=df.loc[base_mask].copy(),
    )


def _coerce_boolean_mask(df: pd.DataFrame, raw_mask: Any, name: str) -> pd.Series:
    if isinstance(raw_mask, pd.DataFrame):
        raise ValueError(f"{name} selector must produce one boolean mask, not a DataFrame.")

    if isinstance(raw_mask, pd.Series):
        mask = raw_mask.copy()
        if not mask.index.equals(df.index):
            raise ValueError(f"{name} selector Series index must match df.index exactly.")
    else:
        array = np.asarray(raw_mask)
        if array.ndim != 1:
            raise ValueError(f"{name} selector must be one-dimensional.")
        if len(array) != len(df):
            raise ValueError(
                f"{name} selector length mismatch: got {len(array)}, expected {len(df)}."
            )
        mask = pd.Series(array, index=df.index)

    if mask.isna().any():
        raise ValueError(f"{name} selector contains NA values.")

    if not pd.api.types.is_bool_dtype(mask):
        unique_values = set(mask.dropna().unique().tolist())
        if unique_values <= {True, False}:
            mask = mask.astype(bool)
        else:
            raise ValueError(f"{name} selector must be boolean, got dtype {mask.dtype}.")

    return mask.astype(bool)
