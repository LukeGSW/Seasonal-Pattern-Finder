# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
pattern_scanner.py – Core engine di scansione dei pattern stagionali (ex Cella 4).
La logica di calcolo del CompositeScore e dell'ordinamento è replicata fedelmente.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

from core.calculations import day_of_year_to_str


def find_seasonal_patterns(
    historical_data_df: pd.DataFrame,
    pivot: pd.DataFrame,
    min_len: int,
    max_len: int,
    min_win_rate: float,
    years_back_value: int,
    top_n_for_print: int,
    min_sharpe: float = 0.0,
) -> pd.DataFrame:
    """
    Cerca pattern stagionali robusti calcolando i rendimenti effettivi dai prezzi storici.
    Ordina i risultati per CompositeScore = 0.2*WinRate_norm + 0.2*MedianReturn_norm + 0.6*SharpeRatio_norm.
    """
    if pivot is None or pivot.empty or historical_data_df is None or historical_data_df.empty:
        print("[-] Dati storici o tabella pivot non validi.", file=sys.stderr)
        return pd.DataFrame()

    results: list[dict] = []

    print(f"[*] Avvio ricerca pattern stagionali:")
    print(f"    Durata pattern: {min_len}-{max_len} gg.")
    print(f"    Win Rate Minimo: {min_win_rate:.1f}%.")
    print(f"    Anni Occorrenze Richiesti: {years_back_value}.")

    if not isinstance(historical_data_df.index, pd.DatetimeIndex):
        historical_data_df.index = pd.to_datetime(historical_data_df.index)

    years = pivot.columns.tolist()
    min_day_idx, max_day_idx = 1, 366

    for start_day in range(min_day_idx, max_day_idx + 1):
        for length in range(min_len, max_len + 1):
            end_day = start_day + length - 1
            if end_day > max_day_idx:
                continue

            annual_returns = {}
            for year in years:
                try:
                    start_date_target = datetime(year, 1, 1) + timedelta(days=start_day - 2)
                    end_date_target = datetime(year, 1, 1) + timedelta(days=end_day - 1)

                    start_price_series = historical_data_df['Adj Close'].loc[
                        start_date_target : start_date_target + timedelta(days=7)
                    ]
                    if start_price_series.empty:
                        continue
                    start_price = start_price_series.iloc[0]

                    end_price_series = historical_data_df['Adj Close'].loc[
                        end_date_target : end_date_target + timedelta(days=7)
                    ]
                    if end_price_series.empty:
                        continue
                    end_price = end_price_series.iloc[0]

                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        annual_returns[year] = (end_price / start_price) - 1
                except Exception:
                    continue

            if len(annual_returns) < years_back_value:
                continue

            valid_returns = pd.Series(annual_returns)
            if valid_returns.empty:
                continue

            wr_long_actual = (valid_returns > 0).mean() * 100
            wr_short_actual = (valid_returns < 0).mean() * 100
            pattern_direction = "LONG" if wr_long_actual >= wr_short_actual else "SHORT"
            current_pattern_win_rate = (
                wr_long_actual if pattern_direction == "LONG"
                else (valid_returns < 0).mean() * 100
            )

            if current_pattern_win_rate < min_win_rate:
                continue

            avg_return = valid_returns.mean()
            median_return = valid_returns.median()
            vol = valid_returns.std(ddof=0)
            sharpe = avg_return / vol if (vol != 0 and not pd.isna(vol) and vol > 1e-9) else np.nan

            pos_sum = valid_returns[valid_returns > 0].sum()
            neg_sum_abs = abs(valid_returns[valid_returns < 0].sum())
            pf = (pos_sum / neg_sum_abs) if neg_sum_abs > 1e-9 else (np.inf if pos_sum > 1e-9 else 1.0)

            results.append({
                "StartDay": start_day,
                "EndDay": end_day,
                "LenDays": length,
                "StartStr": day_of_year_to_str(start_day),
                "EndStr": day_of_year_to_str(end_day),
                "Direction": pattern_direction,
                "WinRate": current_pattern_win_rate,
                "AvgReturn": avg_return,
                "MedianReturn": median_return,
                "Volatility": vol,
                "SharpeRatio": sharpe,
                "ProfitFactor": pf,
                "NYears": len(valid_returns),
                "Returns": valid_returns,
            })

    if not results:
        print("[-] Nessun pattern trovato con i criteri specificati.")
        return pd.DataFrame()

    patterns_df = pd.DataFrame(results)

    # --- CompositeScore (replica esatta) ---
    patterns_df_for_score = patterns_df.copy()
    if 'Direction' in patterns_df_for_score.columns:
        patterns_df_for_score['MedianReturn_score'] = np.where(
            patterns_df_for_score['Direction'] == 'SHORT',
            -patterns_df_for_score['MedianReturn'],
            patterns_df_for_score['MedianReturn'],
        )
        patterns_df_for_score['SharpeRatio_score'] = np.where(
            patterns_df_for_score['Direction'] == 'SHORT',
            -patterns_df_for_score['SharpeRatio'],
            patterns_df_for_score['SharpeRatio'],
        )
    else:
        patterns_df_for_score['MedianReturn_score'] = patterns_df_for_score['MedianReturn']
        patterns_df_for_score['SharpeRatio_score'] = patterns_df_for_score['SharpeRatio']

    patterns_df['WinRate_norm'] = patterns_df['WinRate'].rank(pct=True, na_option='bottom', ascending=True)
    patterns_df['MedianReturn_norm'] = patterns_df_for_score['MedianReturn_score'].rank(
        pct=True, na_option='bottom', ascending=True
    )
    patterns_df['SharpeRatio_norm'] = patterns_df_for_score['SharpeRatio_score'].rank(
        pct=True, na_option='bottom', ascending=True
    )

    weights_score = {'WinRate_norm': 0.2, 'MedianReturn_norm': 0.2, 'SharpeRatio_norm': 0.6}
    patterns_df['CompositeScore'] = (
        weights_score['WinRate_norm'] * patterns_df['WinRate_norm'].fillna(0)
        + weights_score['MedianReturn_norm'] * patterns_df['MedianReturn_norm'].fillna(0)
        + weights_score['SharpeRatio_norm'] * patterns_df['SharpeRatio_norm'].fillna(0)
    )
    patterns_df.sort_values(by="CompositeScore", ascending=False, inplace=True, na_position='last')

    cols_to_drop_after_score = [
        'MedianReturn_score', 'SharpeRatio_score',
        'WinRate_norm', 'MedianReturn_norm', 'SharpeRatio_norm',
    ]
    patterns_df.drop(
        columns=[col for col in cols_to_drop_after_score if col in patterns_df.columns],
        inplace=True,
    )

    return patterns_df
