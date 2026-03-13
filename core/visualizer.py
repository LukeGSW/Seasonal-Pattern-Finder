# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
visualizer.py – Funzioni di visualizzazione (Dark Theme Bloomberg/TradingView).
Ogni funzione restituisce plt.Figure | None (o go.Figure per Plotly).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from core.calculations import calculate_pivot_table, day_of_year_to_str
from core.config import DARK, PLOTLY_LAYOUT


# ─────────────────────────────────────────────────────────────
# Helper interni
# ─────────────────────────────────────────────────────────────
_MONTH_TICK_DOY: list[int] = []
_MONTH_TICK_LABELS: list[str] = []

def _month_ticks():
    """Calcola posizioni e etichette dei mesi sull'asse giorno-dell'anno."""
    global _MONTH_TICK_DOY, _MONTH_TICK_LABELS
    if not _MONTH_TICK_DOY:
        dummy = pd.date_range("2023-01-01", periods=365)
        _MONTH_TICK_DOY = dummy[dummy.is_month_start].dayofyear.tolist()
        _MONTH_TICK_LABELS = dummy[dummy.is_month_start].strftime("%b").tolist()
    return _MONTH_TICK_DOY, _MONTH_TICK_LABELS


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Applica stile uniforme a un asse matplotlib."""
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=DARK["text_bright"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=DARK["text_muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=DARK["text_muted"])
    ax.tick_params(axis='both', labelsize=8, colors=DARK["text_muted"])
    ax.grid(True, linestyle="--", alpha=0.3, color=DARK["grid"])
    for spine in ax.spines.values():
        spine.set_color(DARK["grid"])
        spine.set_linewidth(0.5)


def _apply_month_xticks(ax):
    """Applica etichette dei mesi sull'asse X (giorno dell'anno 1-365)."""
    doy, labels = _month_ticks()
    ax.set_xticks(doy)
    ax.set_xticklabels(labels, fontsize=7, color=DARK["text_muted"])
    ax.set_xlim(1, 365)


# ─────────────────────────────────────────────────────────────
# Utility per report HTML (INVARIATA nella logica)
# ─────────────────────────────────────────────────────────────
def fig_to_base64_html(fig, close_fig: bool = True) -> str:
    """Converte una figura Matplotlib in stringa HTML <img> base64."""
    if fig is None:
        return "<p><i>Grafico non disponibile.</i></p>"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        if close_fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return (
            f'<img src="data:image/png;base64,{img_base64}" alt="grafico" '
            f'style="max-width:100%; height:auto; border:1px solid #30363d; '
            f'border-radius:6px; margin:8px 0;">'
        )
    except Exception as e:
        if fig and close_fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        return f"<p><i>Errore generazione immagine: {e}</i></p>"


# ═════════════════════════════════════════════════════════════
# GRAFICI MATPLOTLIB (Dark Theme)
# ═════════════════════════════════════════════════════════════

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
    values_pct = returns_series.values * 100
    avg_pct = avg_return_pattern * 100
    colors = [DARK["long_green"] if v >= 0 else DARK["short_red"] for v in values_pct]

    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.5), 4.5))
    bars = ax.bar(years.astype(str), values_pct, color=colors, alpha=0.85, edgecolor='none',
                  width=0.7, zorder=3)
    ax.axhline(avg_pct, color=DARK["accent_cyan"], linestyle='--', lw=1.8, zorder=4,
               label=f'Media {avg_pct:.1f}%',
               path_effects=[pe.withStroke(linewidth=3, foreground=DARK["bg_panel"])])
    ax.axhline(0, color=DARK["text_muted"], linewidth=0.6, linestyle=':', zorder=2)

    _style_ax(ax, f"Rendimenti Annuali – {start_str} → {end_str}",
              "Anno", "Rendimento (%)")
    ax.tick_params(axis='x', labelrotation=50, labelsize=7)
    ax.legend(fontsize=8, loc='best')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + (0.8 if h >= 0 else -1.2),
                f'{h:.1f}%', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=6, color=DARK["text_muted"], fontweight='bold')
    fig.tight_layout()
    return fig


def plot_calendar_heatmap(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Calendar Heatmap dei rendimenti medi giornalieri – dark."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        dr = df_hist["Adj Close"].pct_change().dropna() * 100
        if dr.empty:
            return None
        if not isinstance(dr.index, pd.DatetimeIndex):
            dr.index = pd.to_datetime(dr.index)
        avg = dr.groupby([dr.index.month, dr.index.day]).mean()
        hm = avg.unstack(level=0)
        month_map = {i: pd.Timestamp(f'2023-{i}-01').strftime('%b') for i in range(1, 13)}
        hm.columns = hm.columns.map(month_map)
        ordered = [month_map[i] for i in range(1, 13) if month_map[i] in hm.columns]
        hm = hm[ordered]
        if hm.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6.5))
        sns.heatmap(
            hm, cmap="RdYlGn", center=0, annot=False, linewidths=0.3,
            linecolor=DARK["grid"],
            cbar_kws={"format": "%.1f%%", "label": "Rend. Medio (%)",
                       "shrink": 0.8},
            ax=ax,
        )
        # Colorbar text color
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color(DARK["text_muted"])
        cbar.ax.tick_params(colors=DARK["text_muted"])

        _style_ax(ax, f"Calendar Heatmap – {ticker}", "Mese", "Giorno del Mese")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9, color=DARK["text_muted"])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8, color=DARK["text_muted"])
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_polar_seasonality(pivot: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Grafico polare della stagionalità media – dark."""
    if pivot is None or pivot.empty:
        return None
    try:
        avg_dr = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        cum = (1 + avg_dr).cumprod() * 100 - 100
        theta = np.linspace(0, 2 * np.pi, 366, endpoint=False)
        r = cum.loc[1:366].values

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(DARK["bg_panel"])

        ax.plot(theta, r, lw=2, color=DARK["accent_blue"], zorder=3)
        ax.fill(theta, r, alpha=0.15, color=DARK["accent_blue"])
        # Glow effect
        ax.plot(theta, r, lw=4, color=DARK["accent_blue"], alpha=0.15, zorder=2)

        ax.set_title(f"Stagionalità Polare – {ticker}", va='bottom', fontsize=11,
                     fontweight='bold', color=DARK["text_bright"], pad=14)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        month_angles = np.linspace(0, 2 * np.pi, 13)[:-1]
        month_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax.set_xticks(month_angles)
        ax.set_xticklabels(month_labels, fontsize=8, color=DARK["text_muted"])
        ax.tick_params(axis='y', labelsize=7, colors=DARK["text_muted"])

        mn, mx = np.nanmin(r), np.nanmax(r)
        p_mn = np.nanmin([0, mn - abs(mn * 0.1)]) if pd.notna(mn) else 0
        p_mx = np.nanmax([0, mx + abs(mx * 0.1)]) if pd.notna(mx) else 0.1
        if abs(p_mx - p_mn) < 1e-5:
            p_mx = p_mn + 0.1
        ax.set_rmin(p_mn)
        ax.set_rmax(p_mx)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax.grid(True, alpha=0.25, linestyle=':', color=DARK["grid"])
        return fig
    except Exception:
        return None


def plot_monthly_box(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Box Plot dei rendimenti mensili – dark."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        mr = df_hist["Adj Close"].resample('ME').ffill().pct_change().dropna() * 100
        if mr.empty:
            return None
        months = mr.index.month
        fig, ax = plt.subplots(figsize=(11, 5))

        bp = ax.boxplot(
            [mr[months == m].values for m in range(1, 13)],
            patch_artist=True, widths=0.6, showfliers=True,
            flierprops=dict(marker='o', markersize=3, markerfacecolor=DARK["text_muted"], alpha=0.5),
            medianprops=dict(color=DARK["accent_gold"], linewidth=2),
            whiskerprops=dict(color=DARK["text_muted"], linewidth=1),
            capprops=dict(color=DARK["text_muted"], linewidth=1),
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(DARK["palette"][i % len(DARK["palette"])])
            patch.set_alpha(0.65)
            patch.set_edgecolor(DARK["grid"])

        month_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, fontsize=9, color=DARK["text_muted"])
        ax.axhline(0, color=DARK["text_muted"], linewidth=0.6, linestyle=':', zorder=1)
        _style_ax(ax, f"Distribuzione Rendimenti Mensili – {ticker}", "Mese", "Rendimento (%)")
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_stacked_patterns(pats_df: pd.DataFrame, pivot: pd.DataFrame, top_n: int = 5) -> plt.Figure | None:
    """Stacked Area Plot – dark."""
    if pats_df.empty or pivot is None or pivot.empty:
        return None
    num = min(top_n, len(pats_df))
    if num == 0:
        return None
    try:
        plot_pats = pats_df.head(num)
        base = np.zeros(366)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        avg_series = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        count = 0
        for i, (_, pr) in enumerate(plot_pats.iterrows()):
            sd, ed = pr.get('StartDay'), pr.get('EndDay')
            ar = pr.get('AvgReturn', 0.0)
            if pd.isna(sd) or pd.isna(ed):
                continue
            sd, ed = int(sd), int(ed)
            if not (1 <= sd <= 366 and 1 <= ed <= 366 and sd <= ed):
                continue
            seg = avg_series.loc[sd:ed]
            if seg.empty:
                continue
            cum = (1 + seg).cumprod() - 1
            full = np.zeros(366)
            ix_s, ix_e = sd - 1, ed - 1
            if len(cum.values) == (ix_e - ix_s + 1):
                full[ix_s:ix_e + 1] = cum.values
            else:
                continue
            lbl = f"{pr.get('StartStr','NA')}→{pr.get('EndStr','NA')} ({ar:+.1%})" if pd.notna(ar) else "N/A"
            c = DARK["palette"][i % len(DARK["palette"])]
            ax.fill_between(range(1, 367), base * 100, (base + full) * 100,
                            color=c, alpha=0.6, label=lbl, linewidth=0)
            ax.plot(range(1, 367), (base + full) * 100, color=c, alpha=0.8, lw=0.8)
            base += full
            count += 1

        if count == 0:
            plt.close(fig)
            return None

        _style_ax(ax, f"Performance Cumulativa Stacked – Top {count} Pattern", "", "Perf. Media (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        _apply_month_xticks(ax)
        ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.8)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_yearly_overlay(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Overlay performance cumulative annuali – dark."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        pivot = calculate_pivot_table(df_hist)
        if pivot is None or pivot.empty:
            return None
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ny = len(pivot.columns)
        cmap = plt.cm.plasma(np.linspace(0.15, 0.85, ny))
        count = 0
        for i, yc in enumerate(pivot.columns):
            ys = pivot[yc].fillna(0)
            if not ys.empty:
                cp = (1 + ys).cumprod() * 100 - 100
                ax.plot(cp.index, cp.values, alpha=0.35, lw=0.7, color=cmap[i],
                        label=str(yc) if ny <= 12 else None)
                count += 1
        if count == 0:
            plt.close(fig)
            return None
        avg = pivot.mean(axis=1).fillna(0).reindex(range(1, 366), fill_value=0)
        avg_cum = (1 + avg.loc[1:365]).cumprod() * 100 - 100
        ax.plot(avg_cum.index, avg_cum.values, color=DARK["accent_gold"], lw=2.5, ls='--',
                label='Media Annuale', zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground=DARK["bg_panel"])])
        ax.axhline(0, color=DARK["text_muted"], lw=0.5, alpha=0.5)

        _style_ax(ax, f"Overlay Annuale – {ticker}", "", "Perf. da Inizio Anno (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        _apply_month_xticks(ax)
        if ny <= 12:
            ax.legend(ncol=max(1, ny // 4), fontsize=7, loc='best')
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_radar_monthly(df_hist: pd.DataFrame, ticker: str) -> plt.Figure | None:
    """Radar rendimenti medi mensili – dark."""
    if df_hist is None or df_hist.empty:
        return None
    try:
        mr = df_hist["Adj Close"].resample('ME').ffill().pct_change().dropna() * 100
        if mr.empty:
            return None
        avg = mr.groupby(mr.index.month).mean().reindex(range(1, 13), fill_value=0)
        vals = np.concatenate((avg.values, [avg.values[0]]))
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(DARK["bg_panel"])

        ax.plot(angles, vals, 'o-', linewidth=2, color=DARK["accent_cyan"],
                markersize=5, markerfacecolor=DARK["accent_cyan"], markeredgecolor=DARK["bg_panel"])
        ax.fill(angles, vals, alpha=0.15, color=DARK["accent_cyan"])
        ax.plot(angles, vals, lw=4, color=DARK["accent_cyan"], alpha=0.12)

        month_labels = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%b").tolist()
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, month_labels, fontsize=8, color=DARK["text_muted"])
        ax.set_rlabel_position(90)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax.tick_params(axis='y', labelsize=7, colors=DARK["text_muted"])
        ax.grid(True, alpha=0.25, linestyle=':', color=DARK["grid"])

        mn = np.nanmin(vals)
        if pd.notna(mn) and mn < 0:
            ax.set_ylim(bottom=mn * 1.15)
        ax.set_title(f"Radar Mensile – {ticker}", va='bottom', fontsize=11,
                     fontweight='bold', color=DARK["text_bright"], pad=14)
        return fig
    except Exception:
        return None


def plot_seasonal_pattern(
    df_hist: pd.DataFrame,
    ticker: str,
    pivot: pd.DataFrame | None = None,
    patterns_df: pd.DataFrame | None = None,
    top_n: int = 5,
) -> plt.Figure | None:
    """Curva stagionale media cumulativa con highlight pattern – dark."""
    if df_hist is None or df_hist.empty or pivot is None or pivot.empty:
        return None
    try:
        avg = pivot.mean(axis=1).fillna(0).reindex(range(1, 367), fill_value=0)
        cum = (1 + avg).cumprod() * 100 - 100
        fig, ax = plt.subplots(figsize=(14, 6))

        # Linea principale con glow
        ax.plot(cum.index, cum.values, color=DARK["accent_blue"], lw=2.2, label="Stagionale Medio",
                zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground=DARK["bg_panel"])])

        count = 0
        if patterns_df is not None and not patterns_df.empty:
            for idx, pr in patterns_df.head(min(top_n, len(patterns_df))).iterrows():
                sd, ed = pr.get('StartDay'), pr.get('EndDay')
                ar = pr.get('AvgReturn', 0.0)
                if pd.isna(sd) or pd.isna(ed):
                    continue
                sd, ed = int(sd), int(ed)
                c = DARK["long_green_bg"] if ar > 0 else DARK["short_red_bg"]
                if 1 <= sd <= 366 and 1 <= ed <= 366 and sd <= ed:
                    lbl = (f"{pr.get('StartStr','NA')}→{pr.get('EndStr','NA')} ({ar:+.1%})"
                           if count < 4 else "_nolegend_")
                    ax.axvspan(sd, ed + 1, color=c, alpha=0.25, label=lbl, zorder=1)
                    count += 1

        ax.axhline(0, color=DARK["text_muted"], lw=0.6, ls=':', zorder=2)
        _style_ax(ax, f"Stagionalità Media Cumulativa – {ticker}", "", "Performance (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        _apply_month_xticks(ax)
        if count > 0:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════
# EQUITY & DRAWDOWN – Matplotlib (per report HTML)
# ═════════════════════════════════════════════════════════════
def plot_ensemble_equity_and_drawdown(
    df_equity: pd.DataFrame,
    ensemble_name: str,
    selected_patterns_df: pd.DataFrame | None = None,
) -> plt.Figure | None:
    """Equity + drawdown matplotlib – usata per export HTML."""
    if df_equity.empty:
        return None
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Performance Ensemble: {ensemble_name}", fontsize=14,
                 fontweight='bold', color=DARK["text_bright"], y=0.97)
    df_equity['Date'] = pd.to_datetime(df_equity['Date'])

    axs[0].plot(df_equity['Date'], df_equity['CumulativePnL'],
                color=DARK["accent_blue"], lw=2, zorder=10, label='Equity Aggregata')
    axs[0].fill_between(df_equity['Date'], 0, df_equity['CumulativePnL'],
                        alpha=0.08, color=DARK["accent_blue"])

    if selected_patterns_df is not None and not selected_patterns_df.empty:
        mn_yr = df_equity['Date'].min().year
        mx_yr = df_equity['Date'].max().year
        seen = set()
        for yr in range(mn_yr, mx_yr + 1):
            for _, pat in selected_patterns_df.iterrows():
                try:
                    sd = int(pat['StartDay'])
                    ed = int(pat['EndDay'])
                    s_dt = datetime(yr, 1, 1) + timedelta(days=sd - 1)
                    e_yr = yr if ed >= sd else yr + 1
                    e_dt = datetime(e_yr, 1, 1) + timedelta(days=ed - 1)
                    d = pat.get('Direction', 'LONG')
                    c = DARK["long_green"] if d.upper() == 'LONG' else DARK["short_red"]
                    lt = f"{pat.get('StartStr','')}→{pat.get('EndStr','')} ({d})"
                    if lt not in seen:
                        axs[0].axvspan(s_dt, e_dt, color=c, alpha=0.12, label=lt, zorder=1)
                        seen.add(lt)
                    else:
                        axs[0].axvspan(s_dt, e_dt, color=c, alpha=0.12, zorder=1)
                except (ValueError, TypeError):
                    continue

    _style_ax(axs[0], "Equity Curve con Pattern Attivi", "", "P&L ($)")
    axs[0].legend(loc='upper left', fontsize=7, framealpha=0.8)

    peak = df_equity['CumulativePnL'].cummax()
    dd = df_equity['CumulativePnL'] - peak
    axs[1].fill_between(df_equity['Date'], dd, 0, color=DARK["short_red"], alpha=0.55, label='Drawdown ($)')
    axs[1].plot(df_equity['Date'], dd, color=DARK["short_red"], lw=0.8, alpha=0.7)
    _style_ax(axs[1], "", "Data", "Drawdown ($)")
    axs[1].legend(loc='lower right', fontsize=7)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ═════════════════════════════════════════════════════════════
# EQUITY & DRAWDOWN – Plotly Interattivo (per Streamlit)
# ═════════════════════════════════════════════════════════════
def plot_ensemble_equity_plotly(
    df_equity: pd.DataFrame,
    ensemble_name: str,
    selected_patterns_df: pd.DataFrame | None = None,
) -> "go.Figure | None":
    """Equity + drawdown interattivo Plotly (zoom, hover, pan)."""
    if not HAS_PLOTLY or df_equity.empty:
        return None

    df_eq = df_equity.copy()
    df_eq['Date'] = pd.to_datetime(df_eq['Date'])
    peak = df_eq['CumulativePnL'].cummax()
    df_eq['Drawdown'] = df_eq['CumulativePnL'] - peak
    df_eq['DrawdownPct'] = (df_eq['Drawdown'] / peak.replace(0, np.nan) * 100).fillna(0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=["Equity Curve", "Drawdown"],
    )

    # ── Equity ──
    fig.add_trace(
        go.Scatter(
            x=df_eq['Date'], y=df_eq['CumulativePnL'],
            mode='lines', name='Equity',
            line=dict(color=DARK["accent_blue"], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(88,166,255,0.08)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>P&L: $%{y:,.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # ── Pattern Overlay ──
    if selected_patterns_df is not None and not selected_patterns_df.empty:
        mn_yr = df_eq['Date'].min().year
        mx_yr = df_eq['Date'].max().year
        seen_labels = set()
        for yr in range(mn_yr, mx_yr + 1):
            for _, pat in selected_patterns_df.iterrows():
                try:
                    sd = int(pat['StartDay'])
                    ed = int(pat['EndDay'])
                    s_dt = datetime(yr, 1, 1) + timedelta(days=sd - 1)
                    e_yr = yr if ed >= sd else yr + 1
                    e_dt = datetime(e_yr, 1, 1) + timedelta(days=ed - 1)
                    d = pat.get('Direction', 'LONG')
                    c = 'rgba(63,185,80,0.12)' if d.upper() == 'LONG' else 'rgba(248,81,73,0.12)'
                    lt = f"{pat.get('StartStr','')}→{pat.get('EndStr','')} ({d})"
                    show = lt not in seen_labels
                    seen_labels.add(lt)
                    fig.add_vrect(
                        x0=s_dt, x1=e_dt, fillcolor=c, layer="below",
                        line_width=0, row=1, col=1,
                        annotation_text=lt if show and yr == mn_yr else None,
                        annotation_position="top left",
                        annotation_font_size=8,
                        annotation_font_color=DARK["text_muted"],
                    )
                except (ValueError, TypeError):
                    continue

    # ── Drawdown ──
    fig.add_trace(
        go.Scatter(
            x=df_eq['Date'], y=df_eq['Drawdown'],
            mode='lines', name='Drawdown',
            line=dict(color=DARK["short_red"], width=1),
            fill='tozeroy',
            fillcolor='rgba(248,81,73,0.2)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>DD: $%{y:,.2f}<extra></extra>',
        ),
        row=2, col=1,
    )

    # ── Layout ──
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=620,
        title_text=f"Performance Ensemble – {ensemble_name}",
        showlegend=True,
        hovermode='x unified',
        xaxis2=dict(gridcolor=DARK["grid"], zerolinecolor=DARK["grid"]),
        yaxis=dict(title="P&L ($)", gridcolor=DARK["grid"], zerolinecolor=DARK["grid"]),
        yaxis2=dict(title="Drawdown ($)", gridcolor=DARK["grid"], zerolinecolor=DARK["grid"]),
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.color = DARK["text_bright"]
        ann.font.size = 13

    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(count=5, label="5a", step="year", stepmode="backward"),
                dict(step="all", label="Tutto"),
            ],
            font=dict(color=DARK["text"], size=10),
            bgcolor=DARK["bg_card"],
            bordercolor=DARK["grid"],
        ),
    )

    return fig
