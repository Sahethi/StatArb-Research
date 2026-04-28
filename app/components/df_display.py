"""Helpers for displaying DataFrames without column truncation."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _col_width(series: pd.Series) -> str:
    """Pick a Streamlit column width bucket based on max rendered length."""
    try:
        if pd.api.types.is_numeric_dtype(series):
            sample = series.dropna()
            if sample.empty:
                return "small"
            max_abs = float(sample.abs().max())
            if max_abs >= 1e9:
                return "large"
            if max_abs >= 1e5:
                return "medium"
            return "small"
        max_len = int(series.astype(str).str.len().max() or 0)
        if max_len > 25:
            return "large"
        if max_len > 12:
            return "medium"
        return "small"
    except Exception:
        return "medium"


def show_df(df: pd.DataFrame, **kwargs) -> None:
    """
    `st.dataframe` wrapper that sizes columns so large numbers aren't
    truncated to "950..." in the default narrow column.

    Numeric columns get a thousands-separator format and a width bucket
    chosen from the column's max magnitude. String columns are sized
    from their max rendered length. Caller can still pass `column_config`
    to override per-column.
    """
    underlying = getattr(df, "data", df)  # unwrap Styler if needed
    if not isinstance(underlying, pd.DataFrame):
        kwargs.setdefault("width", "stretch")
        st.dataframe(df, **kwargs)
        return
    cols_source = underlying

    user_cfg = kwargs.pop("column_config", {}) or {}
    column_config: dict = {}
    for col in cols_source.columns:
        if col in user_cfg:
            continue
        s = cols_source[col]
        width = _col_width(s)
        if pd.api.types.is_integer_dtype(s):
            column_config[col] = st.column_config.NumberColumn(
                width=width, format="%d"
            )
        elif pd.api.types.is_float_dtype(s):
            column_config[col] = st.column_config.NumberColumn(
                width=width, format="%.4f"
            )
        else:
            column_config[col] = st.column_config.Column(width=width)
    column_config.update(user_cfg)

    kwargs.setdefault("width", "stretch")
    st.dataframe(df, column_config=column_config, **kwargs)
