# SHAP Drift Analyzer

This project provides a small Python library for diagnosing model score drift with XGBoost SHAP values.

It is designed for cases where a model score becomes noticeably higher or lower for a recent population, product, channel, or manually selected customer group. The library compares an explicit `target` population against an explicit `base` population, then exports an Excel report with score distribution, SHAP attribution, feature profile, and optional bivar analysis.

## What It Does

- Accepts a prepared `pandas.DataFrame`; the library does not read source data for you.
- Loads an XGBoost model from `model_path`.
- Requires explicit `target` and `base` selectors.
- Computes model predictions for all rows.
- Compares `target` vs `base` mean SHAP values.
- Profiles top drift features with mean, missing rate, and percentiles.
- Exports Excel every time.
- Generates bivar only when `label_col` is provided.
- Applies `mature` only as an optional extra filter before bivar.

## Basic Usage: String Selectors

```python
import pandas as pd

from shap_analyzer import build_shap_drift_report

df = pd.read_csv("le1/data2.csv")
df["order_dt"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")

result = build_shap_drift_report(
    df=df,
    model_path="le1/CN_POTF_Le1_Risk_SMS_APP_V1.json",
    target="order_dt >= '2026-05-08'",
    base="order_dt < '2026-05-08'",
    output_xlsx="tests/output/le1_example_report.xlsx",
    label_col="t0_cnt",
    mature="bill_t0_ind == 1",
    score_col="model_score",
    top_n=10,
)

print(result.output_xlsx)
print(result.shap_top.head(10))
```

## Basic Usage: Boolean Masks

```python
import pandas as pd

from shap_analyzer import build_shap_drift_report

df = pd.read_csv("le1/data2.csv")
df["order_dt"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")

target_mask = df["product_code"].eq("C")
base_mask = df["product_code"].eq("J")
mature_mask = df["bill_t0_ind"].eq(1)

result = build_shap_drift_report(
    df=df,
    model_path="le1/CN_POTF_Le1_Risk_SMS_APP_V1.json",
    target=target_mask,
    base=base_mask,
    output_xlsx="tests/output/le1_product_c_vs_j.xlsx",
    label_col="t0_cnt",
    mature=mature_mask,
    score_col="model_score",
)

print(result.output_xlsx)
print(result.shap_top.head(10))
```

There is a runnable example at:

```powershell
python examples\le1_shap_drift_example.py
```

The example writes a timestamped workbook under `tests/output/` to avoid overwriting files that may be open in Excel.

## Public API

```python
build_shap_drift_report(
    *,
    df,
    model_path,
    target,
    base,
    output_xlsx=None,
    label_col=None,
    mature=None,
    score_col=None,
    top_n=20,
    bivar_nbins=5,
    image_dir=None,
)
```

### Required Arguments

- `df`: prepared `pandas.DataFrame`.
- `model_path`: path to an XGBoost model file. The model must expose `feature_names`.
- `target`: population selector for the abnormal group.
- `base`: population selector for the baseline group.

### Optional Arguments

- `output_xlsx`: Excel output path. If omitted, the library writes `shap_drift_report_YYYYMMDD_HHMMSS.xlsx` in the current working directory.
- `label_col`: label column. If provided, bivar sheets and charts are generated.
- `mature`: optional selector applied only before bivar.
- `score_col`: optional existing score column, such as an online model score. The library always computes `model_pred`; `score_col` is added as an extra score profile.
- `top_n`: number of top SHAP drift features used in the SHAP chart and `result.shap_top`.
- `bivar_nbins`: number of bivar bins. Bin edges are created from `base`, then applied to `target`.
- `image_dir`: optional directory for generated SHAP and bivar images.

## Selectors

`target`, `base`, and `mature` support two forms.

### String Selector

String selectors are evaluated with `df.eval(..., engine="python")` and must produce a boolean mask.

```python
target = "product_code == 'C' and order_dt >= '2026-05-11'"
base = "product_code == 'C' and order_dt < '2026-05-11'"
```

For date comparisons, prepare a datetime column first:

```python
df["order_dt"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
```

More examples:

```python
target = "product_code == 'C'"
base = "product_code == 'J'"

target = "model_score < 0.03"
base = "model_score >= 0.03"

target = "channel == 'app' and product_code in ['C', 'D']"
base = "channel == 'app' and product_code not in ['C', 'D']"
```

### Boolean Mask

Use boolean masks when the population logic is already computed upstream.

```python
target_mask = df["product_code"].eq("C")
base_mask = df["product_code"].eq("J")

result = build_shap_drift_report(
    df=df,
    model_path="model.json",
    target=target_mask,
    base=base_mask,
)
```

The mask length and index must match `df`.

## Excel Output

Every call writes an Excel report. Sheets include:

- `overview`: all / target / base sample counts and score summary.
- `score_profile`: distribution stats for `model_pred` and optional `score_col`.
- `shap_diff`: SHAP chart and all-feature SHAP detail. The Top table is intentionally not written to Excel.
- `feature_profile`: feature mean, missing rate, and percentiles ordered by SHAP drift.
- `bivar`: only when `label_col` is provided.
- `config`: run parameters and selectors.

Generated images are written next to the Excel file by default, in a folder named after the workbook stem.

## Bivar Behavior

Bivar is disabled unless `label_col` is passed.

```python
result = build_shap_drift_report(
    df=df,
    model_path="model.json",
    target="order_dt >= '2026-05-08'",
    base="order_dt < '2026-05-08'",
    label_col="t0_cnt",
)
```

If `mature` is provided, it is applied after target/base selection and only for bivar:

```python
result = build_shap_drift_report(
    df=df,
    model_path="model.json",
    target="order_dt >= '2026-05-08'",
    base="order_dt < '2026-05-08'",
    label_col="t0_cnt",
    mature="bill_t0_ind == 1",
)
```

If target or base becomes empty after applying `mature`, the function raises `ValueError`.

When bivar is enabled, it is generated for all model features, ordered by absolute SHAP drift. It is not limited by `top_n`.

## Result Object

The function returns `ShapDriftResult`:

```python
result.overview
result.score_profile
result.shap_detail
result.shap_top
result.feature_profile
result.bivar_tables
result.config
result.output_xlsx
result.shap_chart_path
result.bivar_image_paths
```

These are normal pandas objects and paths, so they can be inspected directly in notebooks.

## Validation Rules

The library raises `ValueError` when:

- `df` is empty or not a DataFrame.
- `target` or `base` is missing.
- a selector cannot be evaluated.
- a selector does not produce a boolean mask.
- target/base is empty.
- target/base overlap.
- the model has no `feature_names`.
- `df` is missing model features.
- `score_col` or `label_col` does not exist.
- bivar has no target/base rows after `mature`.

## Tests

Smoke tests use `le1/data2.csv` and `le1/CN_POTF_Le1_Risk_SMS_APP_V1.json`.

```powershell
python -m py_compile shap_bivar_report.py shap_analyzer\*.py tests\test_shap_drift_report.py examples\le1_shap_drift_example.py
python tests\test_shap_drift_report.py
```

The test run writes example reports to:

```text
tests/output/
```
