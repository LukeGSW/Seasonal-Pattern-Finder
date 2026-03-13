# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
visualizer.py – Tutte le funzioni di visualizzazione matplotlib/seaborn (ex Cella 5).
Ogni funzione restituisce un oggetto plt.Figure | None.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64

from core.calculations import calculate_pivot_table, day_of_year_to_str


# ─────────────────────────────────────────────────────────────
# Utility per report HTML
# ─────────────────────────────────────────────────────────────
def fig_to_base64_html(fig, close_fig: bool = True) -> str:
    """Converte una figura Matplotlib in stringa HTML <img> base64."""
    if fig is None:
        return "<p><i>Grafico non disponibile (figura None).</i></p>"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        if close_fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return (
            f'<img src="data:image/png;base64,{img_base64}" alt="grafico" '
            f'style="max-width:100%; height:auto; border:1px solid #ccc; '
            f'margin-top:5px; margin-bottom:5px;">'
        )
    except Exception as e:
        if fig and close_fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return f"<p><i>Errore nella generazione dell'immagine: {e}</i></p>"


# ─────────────────────────────────────────────────────────────
# Funzioni di Plot
# ─────────────────────────────────────────────────────────────
def plot_yearly_returns_barchart(pats_df_single_row: pd.DataFrame, top_n: int = 1) -> plt.Figure | None:
    """Grafico a barre dei rendimenti annuali per un singolo pattern."""
    if pats_df_single_row.empty or len(pats_df_single_row) > 1:
        return None
    row = pats_df_single_row.iloc[0]
    start_str = row.get('StartStr', 'N/A')
    end_str = row.get('EndStr', 'N/A')
    returns_series = row.get('Returns', pd.Series(dtype=float))
    avg_return_pattern = row.get('AvgReturn', 0.0)
    if not isinstance(returns_series, pd.Series) or returns_series.empty:
        return None
    try:
        years = returns_series.index.astype(int)
    except Exception:
        return None
    values_percent = returns_series.values * 100
    avg_return_percent = avg_return_pattern * 100
    colors = ['mediumseagreen' if v >= 0 else 'salmon' for v in values_percent]

    fig_yr, ax_yr = plt.subplots(figsize=(max(7, len(years) * 0.45), 4))
    bars = ax_yr.bar(years.astype(str), values_percent, color=colors, alpha=0.75, edgecolor='grey')
    ax_yr.axhline(avg_return_percent, color='dodgerblue', linestyle='--', lw=1.5,
                  label=f'Rend. Medio ({avg_return_percent:.1f}%)')
    ax_yr.axhline(0, color='black', linewidth=0.7, linestyle=':')
    title = f"Rendimenti Annuali: {row.get('Ticker', 'N/A')} {start_str}→{end_str}"
    ax_yr.set_title(title, fontsize=10, pad=8)
    ax_yr.set_ylabel("Rendimento (%)", fontsize=8)
    ax_yr.set_xlabel("Anno", fontsize=8)
    ax_yr.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax_yr.tick_params(axis='y', labelsize=7)
    ax_yr.grid(axis='y', linestyle='--', alpha=0.5)
    ax_yr.legend(fontsize='x-small')
    for bar in bars:
        height = bar.get_height()
        ax_yr.text(
            bar.get_x() + bar.get_width() / 2.,
            height + (1 if height >= 0 else -1.5),
            f'{height:.1f}%',
            ha='center', va='bottom' if height >= 0 else 'top', fontsize=6,
        )
    fig_yr.tight_layout()
    return fig_yr


def plot_calendar_heatmap(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Calendar Heatmap dei rendimenti medi giornalieri."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        daily_returns_percent = df_hist["Adj Close"].pct_change() * 100
        daily_returns_percent = daily_returns_percent.dropna()
        if daily_returns_percent.empty:
            return None
        if not isinstance(daily_returns_percent.index, pd.DatetimeIndex):
            daily_returns_percent.index = pd.to_datetime(daily_returns_percent.index)
        avg_ret_by_month_day = daily_returns_percent.groupby(
            [daily_returns_percent.index.month, daily_returns_percent.index.day]
        ).mean()
        heatmap_data = avg_ret_by_month_day.unstack(level=0)
        month_map = {i: pd.Timestamp(f'2023-{i}-01').strftime('%b') for i in range(1, 13)}
        heatmap_data.columns = heatmap_data.columns.map(month_map)
        ordered_month_names = [month_map[i] for i in range(1, 13) if month_map[i] in heatmap_data.columns]
        heatmap_data = heatmap_data[ordered_month_names]
        if heatmap_data.empty:
            return None
        fig_ch, ax_ch = plt.subplots(figsize=(11, 6))
        sns.heatmap(
            heatmap_data, cmap="RdYlGn", center=0, annot=False, fmt=".1f",
            linewidths=.2, linecolor='white',
            cbar_kws={"format": "%.1f%%", "label": "Rend. Medio Giorn. (%)"},
            ax=ax_ch,
        )
        ax_ch.set_title(f"Calendar Heatmap – {ticker}", fontsize=11)
        ax_ch.set_ylabel("Giorno del Mese", fontsize=9)
        ax_ch.set_xlabel("Mese", fontsize=9)
        ax_ch.set_xticklabels(ax_ch.get_xticklabels(), rotation=0, fontsize=8)
        ax_ch.set_yticklabels(ax_ch.get_yticklabels(), rotation=0, fontsize=8)
        fig_ch.tight_layout()
        return fig_ch
    except Exception:
        return None


def plot_polar_seasonality(pivot: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Grafico polare della stagionalità media cumulativa."""
    if pivot is None or pivot.empty:
        return None
    try:
        avg_daily_ret = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        cumulative_perf_percent = (1 + avg_daily_ret).cumprod() * 100 - 100
        theta = np.linspace(0, 2 * np.pi, 366, endpoint=False)
        r_values = cumulative_perf_percent.loc[1:366].values
        fig_polar = plt.figure(figsize=(6, 6))
        ax_polar = fig_polar.add_subplot(111, projection="polar")
        ax_polar.plot(theta, r_values, lw=1.8, color='darkslateblue')
        ax_polar.fill(theta, r_values, alpha=0.2, color='cornflowerblue')
        ax_polar.set_title(f"Stagionalità Media Cumul. (%) – {ticker}", va='bottom', fontsize=10, pad=12)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        month_angles = np.linspace(0, 2 * np.pi, 13)[:-1]
        month_names_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax_polar.set_xticks(month_angles)
        ax_polar.set_xticklabels(month_names_labels, fontsize=8)
        min_r = np.nanmin(r_values)
        max_r = np.nanmax(r_values)
        plot_min_r = np.nanmin([0, min_r - abs(min_r * 0.1)]) if pd.notna(min_r) else 0
        plot_max_r = np.nanmax([0, max_r + abs(max_r * 0.1)]) if pd.notna(max_r) else 0.1
        if abs(plot_max_r - plot_min_r) < 1e-5:
            plot_max_r = plot_min_r + 0.1
        ax_polar.set_rmin(plot_min_r)
        ax_polar.set_rmax(plot_max_r)
        ax_polar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax_polar.grid(True, alpha=0.4, linestyle=':')
        return fig_polar
    except Exception:
        return None


def plot_monthly_box(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Box Plot dei rendimenti mensili."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        monthly_returns_percent = df_hist["Adj Close"].resample('ME').ffill().pct_change().dropna() * 100
        if monthly_returns_percent.empty:
            return None
        month_numbers = monthly_returns_percent.index.month
        fig_box, ax_box = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=month_numbers, y=monthly_returns_percent.values, palette="viridis", ax=ax_box)
        month_names_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax_box.set_xticks(ticks=range(12))
        ax_box.set_xticklabels(labels=month_names_labels, fontsize=9)
        ax_box.set_ylabel("Rendimento Mensile (%)", fontsize=9)
        ax_box.set_xlabel("Mese", fontsize=9)
        ax_box.tick_params(axis='y', labelsize=8)
        ax_box.set_title(f"Distribuzione Rendimenti Mensili – {ticker}", fontsize=11)
        ax_box.grid(axis="y", linestyle="--", alpha=0.5)
        ax_box.axhline(0, color='black', linewidth=0.7, linestyle=':')
        fig_box.tight_layout()
        return fig_box
    except Exception:
        return None


def plot_stacked_patterns(pats_df: pd.DataFrame, pivot: pd.DataFrame, top_n: int = 5) -> plt.Figure | None:
    """Stacked Area Plot della performance cumulativa media dei top pattern."""
    if pats_df.empty or pivot is None or pivot.empty:
        return None
    num_to_plot = min(top_n, len(pats_df))
    if num_to_plot == 0:
        return None
    try:
        plot_pats_df = pats_df.head(num_to_plot)
        base_performance = np.zeros(366)
        cmap = plt.get_cmap("viridis")
        fig_stack, ax_stack = plt.subplots(figsize=(12, 5.5))
        avg_daily_ret_series = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        plot_count = 0
        colors_list = cmap(np.linspace(0, 0.9, len(plot_pats_df)))
        for i, (_, pattern_row) in enumerate(plot_pats_df.iterrows()):
            start_day_val = pattern_row.get('StartDay')
            end_day_val = pattern_row.get('EndDay')
            avg_ret_pattern = pattern_row.get('AvgReturn', 0.0)
            if pd.isna(start_day_val) or pd.isna(end_day_val):
                continue
            start_day = int(start_day_val)
            end_day = int(end_day_val)
            if not (1 <= start_day <= 366 and 1 <= end_day <= 366 and start_day <= end_day):
                continue
            segment_daily_returns = avg_daily_ret_series.loc[start_day:end_day]
            if segment_daily_returns.empty:
                continue
            segment_cumulative_performance = (1 + segment_daily_returns).cumprod() - 1
            full_year_segment_performance = np.zeros(366)
            idx_start = start_day - 1
            idx_end = end_day - 1
            if len(segment_cumulative_performance.values) == (idx_end - idx_start + 1):
                full_year_segment_performance[idx_start : idx_end + 1] = segment_cumulative_performance.values
            else:
                continue
            label_return_string = f"{avg_ret_pattern:+.1%}" if pd.notna(avg_ret_pattern) else "N/A"
            ax_stack.fill_between(
                range(1, 367),
                base_performance * 100,
                (base_performance + full_year_segment_performance) * 100,
                color=colors_list[i % len(colors_list)],
                alpha=0.7,
                label=f"{pattern_row.get('StartStr','NA')}→{pattern_row.get('EndStr','NA')} ({label_return_string})",
            )
            base_performance += full_year_segment_performance
            plot_count += 1
        if plot_count > 0:
            ax_stack.set_title(f"Perf. Cumul. Media Stacked (Top {plot_count} Pattern)", fontsize=11)
            ax_stack.set_xlim(1, 365)
            ax_stack.set_ylabel("Perf. Cumul. Media (%)", fontsize=9)
            ax_stack.set_xlabel("Giorno dell'Anno", fontsize=9)
            ax_stack.tick_params(axis='both', labelsize=8)
            ax_stack.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
            ax_stack.legend(fontsize="x-small", loc="upper left", ncol=1)
            ax_stack.grid(linestyle="--", alpha=0.4)
            dummy_dates_xticks = pd.date_range("2023-01-01", periods=365)
            month_starts_doy = dummy_dates_xticks[dummy_dates_xticks.is_month_start].dayofyear.tolist()
            month_labels_xticks = dummy_dates_xticks[dummy_dates_xticks.is_month_start].strftime("%b").tolist()
            ax_stack.set_xticks(month_starts_doy)
            ax_stack.set_xticklabels(month_labels_xticks)
            fig_stack.tight_layout()
            return fig_stack
        else:
            plt.close(fig_stack)
            return None
    except Exception:
        return None


def plot_yearly_overlay(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Overlay delle performance cumulative annuali."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        pivot = calculate_pivot_table(df_hist)
        if pivot is None or pivot.empty:
            return None
        fig_overlay, ax_overlay = plt.subplots(figsize=(12, 5.5))
        num_years = len(pivot.columns)
        colors = plt.cm.viridis(np.linspace(0, 1, num_years))
        plotted_years = 0
        for i, year_col in enumerate(pivot.columns):
            year_series = pivot[year_col].fillna(0)
            if not year_series.empty:
                cumulative_perf = (1 + year_series).cumprod() * 100 - 100
                ax_overlay.plot(
                    cumulative_perf.index, cumulative_perf.values,
                    alpha=0.4, lw=0.7,
                    label=str(year_col) if num_years <= 12 else None,
                    color=colors[i],
                )
                plotted_years += 1
        if plotted_years == 0:
            plt.close(fig_overlay)
            return None
        avg_daily_ret = pivot.mean(axis=1).fillna(0).reindex(range(1, 366), fill_value=0)
        avg_cumulative_perf = (1 + avg_daily_ret.loc[1:365]).cumprod() * 100 - 100
        ax_overlay.plot(
            avg_cumulative_perf.index, avg_cumulative_perf.values,
            color='black', lw=2, ls='--', label='Media Annuale',
        )
        ax_overlay.set_title(f"Performance Cumulativa Annuale Sovrapposta – {ticker}", fontsize=11)
        ax_overlay.set_ylabel("Perf. da Inizio Anno (%)", fontsize=9)
        ax_overlay.set_xlabel("Giorno dell'Anno", fontsize=9)
        ax_overlay.tick_params(axis='both', labelsize=8)
        ax_overlay.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax_overlay.grid(linestyle="--", alpha=0.3)
        ax_overlay.set_xlim(1, 365)
        ax_overlay.axhline(0, color='black', lw=0.7, alpha=0.5)
        if num_years <= 12:
            ax_overlay.legend(ncol=max(1, num_years // 4), fontsize="x-small", loc='best')
        fig_overlay.tight_layout()
        return fig_overlay
    except Exception:
        return None


def plot_radar_monthly(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Radar dei rendimenti medi mensili."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        monthly_returns_percent = df_hist["Adj Close"].resample('ME').ffill().pct_change().dropna() * 100
        if monthly_returns_percent.empty:
            return None
        avg_monthly_ret = monthly_returns_percent.groupby(monthly_returns_percent.index.month).mean().reindex(range(1, 13), fill_value=0)
        values = avg_monthly_ret.values
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        fig_radar = plt.figure(figsize=(6, 6))
        ax_radar = fig_radar.add_subplot(111, projection="polar")
        ax_radar.plot(angles, values, 'o-', linewidth=1.5, color='darkcyan')
        ax_radar.fill(angles, values, alpha=0.2, color='darkcyan')
        month_names_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax_radar.set_thetagrids(angles[:-1] * 180 / np.pi, month_names_labels, fontsize=8)
        ax_radar.set_rlabel_position(90)
        ax_radar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax_radar.grid(True, alpha=0.4, linestyle=':')
        min_val = np.nanmin(values)
        if pd.notna(min_val) and min_val < 0:
            ax_radar.set_ylim(bottom=min_val * 1.1)
        ax_radar.set_title(f"Radar Rendimento Medio Mensile (%) – {ticker}", va='bottom', fontsize=10, pad=12)
        return fig_radar
    except Exception:
        return None


def plot_seasonal_pattern(
    df_hist: pd.DataFrame,
    ticker: str,
    pivot: pd.DataFrame | None = None,
    patterns_df: pd.DataFrame | None = None,
    top_n: int = 5,
) -> plt.Figure | None:
    """Grafico stagionale medio cumulativo con evidenziazione dei top pattern."""
    if df_hist is None or df_hist.empty or pivot is None or pivot.empty:
        return None
    try:
        avg_daily_ret = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        cumulative_perf_percent = (1 + avg_daily_ret).cumprod() * 100 - 100
        fig_seasonal, ax_seasonal = plt.subplots(figsize=(12, 5.5))
        ax_seasonal.plot(
            cumulative_perf_percent.index, cumulative_perf_percent.values,
            color="darkblue", lw=1.8, label="Stagionale Medio",
        )
        plotted_highlights_count = 0
        if patterns_df is not None and not patterns_df.empty:
            num_to_highlight = min(top_n, len(patterns_df))
            for idx, pattern_row in patterns_df.head(num_to_highlight).iterrows():
                start_day_pat = pattern_row.get('StartDay')
                end_day_pat = pattern_row.get('EndDay')
                avg_ret_pat = pattern_row.get('AvgReturn', 0.0)
                if pd.isna(start_day_pat) or pd.isna(end_day_pat):
                    continue
                start_day = int(start_day_pat)
                end_day = int(end_day_pat)
                highlight_color = "lightgreen" if avg_ret_pat > 0 else "lightcoral"
                if 1 <= start_day <= 366 and 1 <= end_day <= 366 and start_day <= end_day:
                    label_for_span = (
                        f"{pattern_row.get('StartStr','NA')}→{pattern_row.get('EndStr','NA')} ({avg_ret_pat:+.1%})"
                        if plotted_highlights_count < 3 else "_nolegend_"
                    )
                    ax_seasonal.axvspan(start_day, end_day + 1, color=highlight_color, alpha=0.3, label=label_for_span)
                    plotted_highlights_count += 1
        ax_seasonal.set_title(f"Stagionalità Media Cumulativa – {ticker}", fontsize=11)
        ax_seasonal.set_ylabel("Performance Cumul. Media (%)", fontsize=9)
        ax_seasonal.set_xlabel("Giorno dell'Anno", fontsize=9)
        ax_seasonal.tick_params(axis='both', labelsize=8)
        ax_seasonal.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax_seasonal.axhline(0, color="black", lw=0.7, alpha=0.6, linestyle=':')
        ax_seasonal.grid(linestyle="--", alpha=0.4)
        ax_seasonal.set_xlim(1, 365)
        dummy_dates = pd.date_range("2023-01-01", periods=365)
        month_starts_doy = dummy_dates[dummy_dates.is_month_start].dayofyear.tolist()
        month_labels = dummy_dates[dummy_dates.is_month_start].strftime("%b").tolist()
        ax_seasonal.set_xticks(month_starts_doy)
        ax_seasonal.set_xticklabels(month_labels)
        if plotted_highlights_count > 0:
            ax_seasonal.legend(fontsize="x-small", loc="best")
        fig_seasonal.tight_layout()
        return fig_seasonal
    except Exception:
        return None


def plot_ensemble_equity_and_drawdown(
    df_equity: pd.DataFrame,
    ensemble_name: str,
    selected_patterns_df: pd.DataFrame | None = None,
) -> plt.Figure | None:
    """Plot equity curve e drawdown per il backtest ensemble con overlay dei periodi pattern."""
    if df_equity.empty:
        return None
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Performance Ensemble Pattern: {ensemble_name}", fontsize=16)
    df_equity['Date'] = pd.to_datetime(df_equity['Date'])

    axs[0].plot(
        df_equity['Date'], df_equity['CumulativePnL'],
        label='Equity Curve Aggregata', color='darkblue', lw=2, zorder=10,
    )

    if selected_patterns_df is not None and not selected_patterns_df.empty:
        min_year = df_equity['Date'].min().year
        max_year = df_equity['Date'].max().year
        plotted_labels = set()
        for year in range(min_year, max_year + 1):
            for _, pattern in selected_patterns_df.iterrows():
                try:
                    start_day = int(pattern['StartDay'])
                    end_day = int(pattern['EndDay'])
                    start_date = datetime(year, 1, 1) + timedelta(days=start_day - 1)
                    end_year = year if end_day >= start_day else year + 1
                    end_date = datetime(end_year, 1, 1) + timedelta(days=end_day - 1)
                    direction = pattern.get('Direction', 'LONG')
                    color = 'lightgreen' if direction.upper() == 'LONG' else 'lightcoral'
                    label_text = f"Pattern: {pattern.get('StartStr','')}→{pattern.get('EndStr','')} ({direction})"
                    if label_text not in plotted_labels:
                        axs[0].axvspan(start_date, end_date, color=color, alpha=0.2, label=label_text, zorder=1)
                        plotted_labels.add(label_text)
                    else:
                        axs[0].axvspan(start_date, end_date, color=color, alpha=0.2, zorder=1)
                except (ValueError, TypeError):
                    continue

    axs[0].set_title('Andamento del Valore del Portafoglio Aggregato con Pattern Attivi')
    axs[0].set_ylabel('Valore Portafoglio ($)')
    axs[0].grid(True, linestyle=':')
    axs[0].legend(loc='upper left', fontsize='small')

    peak = df_equity['CumulativePnL'].cummax()
    drawdown = df_equity['CumulativePnL'] - peak
    axs[1].fill_between(df_equity['Date'], drawdown, 0, label='Drawdown ($)', color='salmon', alpha=0.7)
    axs[1].set_ylabel('Drawdown ($)')
    axs[1].set_xlabel('Data')
    axs[1].grid(True, linestyle=':')
    axs[1].legend(loc='lower right', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
