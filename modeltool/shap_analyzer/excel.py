from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bivar import make_bivar_wide_table


def export_excel(
    *,
    overview: pd.DataFrame,
    score_profile: pd.DataFrame,
    shap_detail: pd.DataFrame,
    shap_top: pd.DataFrame,
    feature_profile: pd.DataFrame,
    bivar_tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    output_xlsx: str | Path,
    shap_chart_path: str | Path,
    bivar_image_paths: dict[str, Path],
) -> Path:
    output = Path(output_xlsx)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        formats = build_formats(workbook)

        write_overview_sheet(writer, overview, formats)
        write_score_sheet(writer, score_profile, formats)
        write_shap_sheet(writer, shap_detail, shap_top, Path(shap_chart_path), formats)
        write_feature_profile_sheet(writer, feature_profile, formats)
        if bivar_tables:
            write_bivar_sheet(writer, bivar_tables, bivar_image_paths, formats)
        write_config_sheet(writer, config, formats)

        for worksheet in writer.sheets.values():
            worksheet.set_default_row(19)

    return output


def build_formats(workbook) -> dict[str, Any]:
    font_name = "Microsoft YaHei"
    theme_blue = "#1F4E78"
    theme_lighter = "#EEF4FB"
    theme_border = "#B8C7D9"

    formats = {
        "title": workbook.add_format(
            {
                "font_name": font_name,
                "bold": True,
                "font_size": 16,
                "font_color": "#17375E",
            }
        ),
        "section": workbook.add_format(
            {
                "font_name": font_name,
                "bold": True,
                "font_size": 12,
                "font_color": "#17375E",
                "bg_color": "#EAF2F8",
                "bottom": 1,
                "bottom_color": theme_border,
            }
        ),
        "note": workbook.add_format(
            {
                "font_name": font_name,
                "font_size": 9,
                "font_color": "#5B6573",
                "italic": True,
            }
        ),
        "header": workbook.add_format(
            {
                "font_name": font_name,
                "bold": True,
                "font_color": "white",
                "bg_color": theme_blue,
                "border": 1,
                "border_color": theme_border,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "pos_float": workbook.add_format(
            {
                "font_name": font_name,
                "font_size": 10,
                "bg_color": "#DCE6F1",
                "border": 1,
                "border_color": theme_border,
                "num_format": "0.0000",
                "font_color": "#1F4E78",
            }
        ),
        "neg_float": workbook.add_format(
            {
                "font_name": font_name,
                "font_size": 10,
                "bg_color": "#FCE4D6",
                "border": 1,
                "border_color": theme_border,
                "num_format": "0.0000",
                "font_color": "#9E480E",
            }
        ),
    }

    for row_type, bg_color in [("body", "white"), ("body_alt", theme_lighter)]:
        for col_type, num_format in [
            ("text", None),
            ("int", "#,##0"),
            ("float", "0.0000"),
            ("percent", "0.00%"),
        ]:
            options = {
                "font_name": font_name,
                "font_size": 10,
                "bg_color": bg_color,
                "border": 1,
                "border_color": theme_border,
            }
            if num_format is not None:
                options["num_format"] = num_format
            formats[f"{row_type}_{col_type}"] = workbook.add_format(options)

    return formats


def write_overview_sheet(writer, overview: pd.DataFrame, formats: dict[str, Any]) -> None:
    ws = writer.book.add_worksheet("overview")
    writer.sheets["overview"] = ws
    ws.hide_gridlines(2)
    ws.freeze_panes(3, 0)
    ws.set_zoom(90)
    ws.set_tab_color("#5B9BD5")
    ws.merge_range(0, 0, 0, 8, "SHAP Drift Report", formats["title"])
    ws.write(1, 0, "Sample overview for all / target / base.", formats["note"])
    write_dataframe_formatted(ws, overview, 3, 0, formats, apply_autofilter=False)
    autosize_columns(ws, overview)


def write_score_sheet(writer, score_profile: pd.DataFrame, formats: dict[str, Any]) -> None:
    ws = writer.book.add_worksheet("score_profile")
    writer.sheets["score_profile"] = ws
    ws.hide_gridlines(2)
    ws.freeze_panes(3, 0)
    ws.set_zoom(90)
    ws.set_tab_color("#70AD47")
    ws.merge_range(0, 0, 0, 8, "Score Profile", formats["title"])
    ws.write(1, 0, "Score distribution stats for model_pred and optional score_col.", formats["note"])
    write_dataframe_formatted(ws, score_profile, 3, 0, formats, apply_autofilter=True)
    autosize_columns(ws, score_profile)


def write_shap_sheet(
    writer,
    shap_detail: pd.DataFrame,
    shap_top: pd.DataFrame,
    shap_chart_path: Path,
    formats: dict[str, Any],
) -> None:
    ws = writer.book.add_worksheet("shap_diff")
    writer.sheets["shap_diff"] = ws
    ws.hide_gridlines(2)
    ws.freeze_panes(3, 0)
    ws.set_zoom(88)
    ws.set_tab_color("#4472C4")
    ws.merge_range(0, 0, 0, 8, "Target vs Base SHAP Difference", formats["title"])
    ws.write(1, 0, "Rows are ordered by absolute mean SHAP difference.", formats["note"])
    ws.write(2, 0, "Top SHAP difference chart", formats["section"])
    ws.insert_image(3, 0, str(shap_chart_path), {"x_scale": 0.82, "y_scale": 0.82})

    detail_row = 28
    ws.write(detail_row, 0, "All model features", formats["section"])
    write_dataframe_formatted(ws, shap_detail, detail_row + 1, 0, formats, apply_autofilter=True)
    add_shap_sign_formatting(ws, shap_detail, detail_row + 1, 0, formats)
    autosize_columns(ws, shap_detail)
    ws.set_column(0, 0, 30)


def write_feature_profile_sheet(
    writer, feature_profile: pd.DataFrame, formats: dict[str, Any]
) -> None:
    ws = writer.book.add_worksheet("feature_profile")
    writer.sheets["feature_profile"] = ws
    ws.hide_gridlines(2)
    ws.freeze_panes(3, 1)
    ws.set_zoom(88)
    ws.set_tab_color("#ED7D31")
    ws.merge_range(0, 0, 0, 10, "Feature Profile", formats["title"])
    ws.write(
        1,
        0,
        "Feature mean, missing rate, and percentiles ordered by SHAP drift.",
        formats["note"],
    )
    write_dataframe_formatted(ws, feature_profile, 3, 0, formats, apply_autofilter=True)
    autosize_columns(ws, feature_profile)
    ws.set_column(0, 0, 30)


def write_bivar_sheet(
    writer,
    bivar_tables: dict[str, pd.DataFrame],
    bivar_image_paths: dict[str, Path],
    formats: dict[str, Any],
) -> None:
    ws = writer.book.add_worksheet("bivar")
    writer.sheets["bivar"] = ws
    ws.hide_gridlines(2)
    ws.set_zoom(86)
    ws.set_tab_color("#A5A5A5")
    ws.merge_range(0, 0, 0, 10, "Bivar | Target vs Base", formats["title"])
    ws.write(
        1,
        0,
        "Bins are built from base and applied to target. Line = label rate, bar = population share.",
        formats["note"],
    )
    row_cursor = 3
    for feature, table in bivar_tables.items():
        wide_table = make_bivar_wide_table(table)
        ws.merge_range(row_cursor, 0, row_cursor, 10, feature, formats["section"])
        ws.insert_image(row_cursor + 1, 0, str(bivar_image_paths[feature]), {"x_scale": 0.84, "y_scale": 0.84})
        table_row = row_cursor + 23
        write_dataframe_formatted(ws, wide_table, table_row, 0, formats, apply_autofilter=False)
        row_cursor = table_row + len(wide_table) + 4
    ws.set_column(0, 0, 8)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 12, 14)


def write_config_sheet(writer, config: dict[str, Any], formats: dict[str, Any]) -> None:
    ws = writer.book.add_worksheet("config")
    writer.sheets["config"] = ws
    ws.hide_gridlines(2)
    ws.set_zoom(90)
    ws.set_tab_color("#8064A2")
    ws.merge_range(0, 0, 0, 3, "Run Config", formats["title"])
    config_df = pd.DataFrame(
        [{"key": key, "value": stringify_config_value(value)} for key, value in config.items()]
    )
    write_dataframe_formatted(ws, config_df, 2, 0, formats, apply_autofilter=False)
    ws.set_column(0, 0, 28)
    ws.set_column(1, 1, 90)


def add_shap_sign_formatting(
    worksheet,
    dataframe: pd.DataFrame,
    start_row: int,
    start_col: int,
    formats: dict[str, Any],
) -> None:
    if "target_minus_base" not in dataframe.columns or dataframe.empty:
        return
    col = start_col + dataframe.columns.get_loc("target_minus_base")
    first_row = start_row + 1
    last_row = start_row + len(dataframe)
    worksheet.conditional_format(
        first_row,
        col,
        last_row,
        col,
        {"type": "cell", "criteria": ">=", "value": 0, "format": formats["pos_float"]},
    )
    worksheet.conditional_format(
        first_row,
        col,
        last_row,
        col,
        {"type": "cell", "criteria": "<", "value": 0, "format": formats["neg_float"]},
    )


def autosize_columns(worksheet, dataframe: pd.DataFrame, start_col: int = 0) -> None:
    for col_idx, col_name in enumerate(dataframe.columns, start=start_col):
        series = dataframe[col_name].astype(str)
        max_len = max([len(str(col_name))] + series.map(len).tolist())
        worksheet.set_column(col_idx, col_idx, min(max_len + 2, 28))


def infer_col_type(col_name: str) -> str:
    lower = col_name.lower()
    if (
        "rate" in lower
        or "pct" in lower
        or "share" in lower
        or lower.endswith("_ratio")
    ):
        return "percent"
    if (
        lower in {"grp", "count", "n", "pop_n", "label_n", "total_count"}
        or lower.endswith("_count")
        or lower.endswith("_n")
    ):
        return "int"
    if lower in {"feature", "group_name", "compare_group", "bin_label", "direction", "score_name", "key", "value"}:
        return "text"
    return "float"


def write_dataframe_formatted(
    worksheet,
    dataframe: pd.DataFrame,
    start_row: int,
    start_col: int,
    formats: dict[str, Any],
    apply_autofilter: bool = False,
) -> tuple[int, int]:
    for col_idx, col_name in enumerate(dataframe.columns):
        worksheet.write(start_row, start_col + col_idx, col_name, formats["header"])

    for row_idx, (_, row) in enumerate(dataframe.iterrows(), start=1):
        row_type = "body_alt" if row_idx % 2 == 0 else "body"
        for col_idx, col_name in enumerate(dataframe.columns):
            value = normalize_excel_value(row[col_name])
            col_type = infer_col_type(col_name)
            base_format = formats[f"{row_type}_{col_type}"]
            if value is None:
                worksheet.write_blank(start_row + row_idx, start_col + col_idx, None, base_format)
            else:
                worksheet.write(start_row + row_idx, start_col + col_idx, value, base_format)

    end_row = start_row + len(dataframe)
    end_col = start_col + dataframe.shape[1] - 1
    if apply_autofilter and len(dataframe) > 0:
        worksheet.autofilter(start_row, start_col, end_row, end_col)
    return end_row, end_col


def normalize_excel_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isinf(value):
        return None
    return value


def stringify_config_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)
