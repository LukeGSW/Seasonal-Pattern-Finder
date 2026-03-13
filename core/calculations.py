# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
calculations.py – Funzioni utilità di calcolo: pivot table, day_of_year_to_str,
                   get_shifted_window_avg_return (robustezza).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_pivot_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Calcola la pivot table dei rendimenti giornalieri indicizzata per giorno dell'anno."""
    if df is None or df.empty:
        return None
    try:
        if 'Adj Close' not in df.columns:
            return None
        daily_returns = df["Adj Close"].pct_change().dropna()
        if daily_returns.empty:
            return None
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)
        daily_returns = daily_returns[~daily_returns.index.duplicated(keep='first')]
        pivot = pd.pivot_table(
            daily_returns.to_frame("R"),
            index=daily_returns.index.dayofyear,
            columns=daily_returns.index.year,
            values="R",
        )
        if pivot.empty:
            return None
        pivot = pivot.reindex(range(1, 367))
        return pivot
    except Exception:
        return None


def day_of_year_to_str(day_of_year: int | float) -> str:
    """Converte il giorno dell'anno (1-366) nella stringa 'Mmm-dd'."""
    if pd.isna(day_of_year):
        return "N/A"
    try:
        day_of_year = int(day_of_year)
        base_year_for_display = 2023  # anno non bisestile
        if day_of_year == 366:
            target_date = pd.Timestamp(f"{base_year_for_display}-12-31")
        elif 1 <= day_of_year <= 365:
            base_date = pd.Timestamp(f"{base_year_for_display}-01-01")
            target_date = base_date + pd.Timedelta(days=day_of_year - 1)
        else:
            return f"Giorno {day_of_year}"
        return target_date.strftime("%b-%d")
    except Exception:
        return f"Giorno {day_of_year}"


def get_shifted_window_avg_return(
    pivot_table: pd.DataFrame,
    original_start_day: int,
    original_end_day: int,
    day_offset: int,
    years_to_consider: int,
) -> float | None:
    """
    Calcola il rendimento medio annuale per una finestra stagionale shiftata.
    L'offset negativo anticipa l'inizio, l'offset positivo posticipa la fine.
    """
    if pivot_table is None or pivot_table.empty:
        return None

    shifted_start_day = original_start_day - day_offset
    shifted_end_day = original_end_day + day_offset

    min_pivot_idx = pivot_table.index.min()
    max_pivot_idx = pivot_table.index.max()
    shifted_start_day = max(min_pivot_idx, shifted_start_day)
    shifted_end_day = min(max_pivot_idx, shifted_end_day)

    if shifted_start_day > shifted_end_day:
        return None

    if not (shifted_start_day in pivot_table.index and shifted_end_day in pivot_table.index):
        return None

    try:
        window_daily_returns = pivot_table.loc[shifted_start_day:shifted_end_day]
        current_length = shifted_end_day - shifted_start_day + 1
        min_valid_days_in_window = current_length * 0.5

        def calculate_return_for_year_shifted(year_series):
            valid_days = year_series.dropna()
            if len(valid_days) < min_valid_days_in_window:
                return np.nan
            return (1 + valid_days).prod() - 1

        annual_compounded_returns = window_daily_returns.apply(calculate_return_for_year_shifted, axis=0)
        valid_annual_returns = annual_compounded_returns.dropna()

        if len(valid_annual_returns) < years_to_consider:
            return None

        return valid_annual_returns.mean()
    except Exception:
        return None
