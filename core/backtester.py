# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
backtester.py – Motore di backtest ensemble per portafoglio di pattern (ex Cella 8).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_trade_price_from_ohlc(
    ohlc_df: pd.DataFrame,
    target_date: datetime,
    price_type: str = 'open',
) -> tuple[float | None, datetime | None]:
    """Cerca il prezzo più prossimo a target_date nel DataFrame OHLC."""
    if ohlc_df is None or ohlc_df.empty or price_type not in ohlc_df.columns:
        return None, None
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        try:
            ohlc_df.index = pd.to_datetime(ohlc_df.index)
        except Exception:
            return None, None
    if not ohlc_df.index.is_monotonic_increasing:
        ohlc_df = ohlc_df.sort_index()
    try:
        idx_pos = ohlc_df.index.searchsorted(target_date, side='left')
        if idx_pos < len(ohlc_df.index):
            actual_date = ohlc_df.index[idx_pos]
            price = ohlc_df[price_type].iloc[idx_pos]
            if pd.notna(price) and price > 0:
                return price, actual_date
    except Exception:
        pass
    return None, None


def run_ensemble_backtest_single_ticker(
    selected_patterns: pd.DataFrame,
    ticker_ohlc_data: pd.DataFrame,
    first_backtest_year: int,
    last_backtest_year: int,
    capital_per_full_weight_trade: float,
    ticker_symbol_for_log: str = "N/A",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Esegue il backtest ensemble su un singolo ticker dato un portafoglio di pattern selezionati.
    Restituisce: (df_all_trades, df_annual_pnl, df_portfolio_equity, empty_series).
    """
    all_trades_list: list[dict] = []
    annual_pnl_summary: dict = {}
    daily_pnl_contributions_list: list[dict] = []
    ohlc_index_for_exposure = ticker_ohlc_data.loc[
        f"{first_backtest_year}":f"{last_backtest_year}"
    ].index
    if ohlc_index_for_exposure.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    for year in range(first_backtest_year, last_backtest_year + 1):
        pnl_for_this_year_ensemble = 0.0
        trades_this_year_ensemble = 0
        for _, pattern_row in selected_patterns.iterrows():
            start_day_of_year = int(pattern_row['StartDay'])
            end_day_of_year = int(pattern_row['EndDay'])
            direction = pattern_row['Direction']
            weight = pattern_row['PortfolioWeight']
            if weight <= 1e-6:
                continue
            capital_for_this_instance = capital_per_full_weight_trade * weight
            try:
                date_ref_start_pattern = datetime(year, 1, 1) + timedelta(days=start_day_of_year - 1)
                exit_year_actual_pattern = year
                if end_day_of_year < start_day_of_year:
                    exit_year_actual_pattern = year + 1
                date_ref_end_pattern = datetime(exit_year_actual_pattern, 1, 1) + timedelta(days=end_day_of_year - 1)
                entry_date_target_trade = date_ref_start_pattern - timedelta(days=1)
                exit_trigger_date_for_trade = date_ref_end_pattern
            except ValueError:
                continue

            entry_price, entry_date_actual_trade = get_trade_price_from_ohlc(
                ticker_ohlc_data, entry_date_target_trade, 'Adj Close'
            )
            exit_price, exit_date_actual_trade = get_trade_price_from_ohlc(
                ticker_ohlc_data, exit_trigger_date_for_trade, 'Adj Close'
            )
            if not all([entry_price, entry_date_actual_trade, exit_price, exit_date_actual_trade]):
                continue
            if exit_date_actual_trade <= entry_date_actual_trade:
                continue

            num_shares = capital_for_this_instance / entry_price
            pnl_trade = (
                ((exit_price - entry_price) if direction.upper() == "LONG" else (entry_price - exit_price))
                * num_shares
            )
            pnl_for_this_year_ensemble += pnl_trade
            trades_this_year_ensemble += 1
            all_trades_list.append({
                'Year': year,
                'PnL_Trade': pnl_trade,
                'CapitalAllocated': capital_for_this_instance,
                'EntryDateActual': entry_date_actual_trade,
                'ExitDateActual': exit_date_actual_trade,
                'Pattern': f"{pattern_row.get('StartStr','')}→{pattern_row.get('EndStr','')}",
            })
            daily_pnl_contributions_list.append({
                'Date': exit_date_actual_trade,
                'PnL_Contribution': pnl_trade,
            })
        annual_pnl_summary[year] = {
            'PnL': pnl_for_this_year_ensemble,
            'NumTrades': trades_this_year_ensemble,
        }

    if not all_trades_list:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    df_all_trades_ensemble = pd.DataFrame(all_trades_list)
    df_annual_pnl_ensemble = pd.DataFrame.from_dict(annual_pnl_summary, orient='index')
    df_daily_pnl = pd.DataFrame(daily_pnl_contributions_list)
    if df_daily_pnl.empty:
        return df_all_trades_ensemble, df_annual_pnl_ensemble, pd.DataFrame(), pd.Series(dtype=float)

    df_daily_pnl['Date'] = pd.to_datetime(df_daily_pnl['Date'])
    df_equity_temp_grouped = df_daily_pnl.groupby('Date')['PnL_Contribution'].sum().sort_index()
    df_equity_temp_series = df_equity_temp_grouped.reindex(ohlc_index_for_exposure, fill_value=0.0).cumsum()
    df_temp = df_equity_temp_series.reset_index()
    df_temp.columns = ['Date', 'CumulativePnL']
    df_portfolio_equity_ensemble = df_temp

    return df_all_trades_ensemble, df_annual_pnl_ensemble, df_portfolio_equity_ensemble, pd.Series(dtype=float)
