# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder – Streamlit App (v2 Dark Theme)
app.py – Entry point con layout a tabs, KPI cards e grafici Plotly interattivi.

Fasi:
  1. Sidebar: Input parametri → Avvia Analisi → salva top_df in session_state
  2. Tab "Pattern": KPI cards + tabella + selezione/pesatura con data_editor
  3. Tab "Grafici": tutte le visualizzazioni (metriche, robustezza, dettagli asset)
  4. Tab "Backtest": backtest ensemble con equity Plotly + report HTML scaricabile
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import date, datetime, timedelta
import warnings

# ── Moduli core ──
from core.config import (
    RISK_FREE_RATE_ANNUAL, BACKTEST_CAPITAL_PER_FULL_WEIGHT_TRADE,
    REPORT_HTML_CSS, DARK, PLOTLY_LAYOUT, STREAMLIT_CUSTOM_CSS, apply_dark_theme,
)
from core.data_fetcher import download_data
from core.calculations import calculate_pivot_table, day_of_year_to_str, get_shifted_window_avg_return
from core.pattern_scanner import find_seasonal_patterns
from core.visualizer import (
    fig_to_base64_html,
    plot_seasonal_pattern,
    plot_yearly_returns_barchart,
    plot_calendar_heatmap,
    plot_polar_seasonality,
    plot_monthly_box,
    plot_stacked_patterns,
    plot_yearly_overlay,
    plot_radar_monthly,
    plot_ensemble_equity_and_drawdown,
    plot_ensemble_equity_plotly,
)
from core.backtester import run_ensemble_backtest_single_ticker

# ── Impostazioni globali ──
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue`")
warnings.filterwarnings("ignore", message="'M' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
apply_dark_theme()

# ── Page Config ──
st.set_page_config(
    page_title="KriterionQuant – Seasonal Pattern Finder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(STREAMLIT_CUSTOM_CSS, unsafe_allow_html=True)

# ── Header ──
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>"
    "📈 KriterionQuant <span style='color:#58a6ff;'>Seasonal Pattern Finder</span></h1>"
    "<p style='text-align:center; color:#8b949e; margin-top:4px; font-size:0.9rem;'>"
    "Analisi di stagionalità • Composite Score • Backtest Ensemble</p>",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# Session State Initialization
# ═════════════════════════════════════════════════════════════
for key in [
    "top_df", "historical_data_df", "pivot_table", "ui_params",
    "report_html_parts", "selected_weighted_patterns_df",
    "analysis_done", "selection_confirmed", "backtest_done",
]:
    if key not in st.session_state:
        if key in ("top_df", "historical_data_df", "selected_weighted_patterns_df"):
            st.session_state[key] = pd.DataFrame()
        elif key == "pivot_table":
            st.session_state[key] = None
        elif key == "report_html_parts":
            st.session_state[key] = []
        elif key == "ui_params":
            st.session_state[key] = {}
        else:
            st.session_state[key] = False


# ═════════════════════════════════════════════════════════════
# SIDEBAR – Input Parametri
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:8px 0 12px 0;'>"
        "<span style='font-size:1.4rem; font-weight:700; color:#f0f6fc;'>⚙️ Parametri</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    ticker_val = st.text_input("Ticker (EODHD)", value="BNB-USD.CC", help="Es: AAPL.US, BTC-USD.CC, SPY.US")
    years_back_val = st.slider("Anni Storico", 5, 30, 20)
    col_a, col_b = st.columns(2)
    with col_a:
        min_len_val = st.number_input("Min Durata (gg)", 5, 90, 20, step=1)
    with col_b:
        max_len_val = st.number_input("Max Durata (gg)", 5, 120, 60, step=1)
    min_win_rate_val = st.number_input("Min Win Rate (%)", 50.0, 100.0, 75.0, step=0.5)
    top_n_val = st.slider("Top N Pattern", 1, 15, 15)
    range_robust_val = st.slider("Range Robustezza (±gg)", 0, 7, (0, 3))

    st.markdown("---")
    st.markdown("<p style='font-weight:600; color:#c9d1d9; font-size:0.85rem;'>Visualizzazioni</p>",
                unsafe_allow_html=True)
    show_metrics_plots = st.checkbox("Metriche Chiave Pattern", value=True)
    show_robustness = st.checkbox("Heatmap Robustezza", value=True)
    show_annual_returns = st.checkbox("Rendimenti Annuali", value=True)
    show_detailed_charts = st.checkbox("Grafici Dettagliati Asset", value=True)

    st.markdown("---")
    run_analysis = st.button("🚀 Avvia Analisi", use_container_width=True, type="primary")


# ═════════════════════════════════════════════════════════════
# FASE 1 – Esecuzione Analisi
# ═════════════════════════════════════════════════════════════
if run_analysis:
    st.session_state.analysis_done = False
    st.session_state.selection_confirmed = False
    st.session_state.backtest_done = False
    st.session_state.top_df = pd.DataFrame()
    st.session_state.selected_weighted_patterns_df = pd.DataFrame()
    st.session_state.report_html_parts = []

    ticker_upper = ticker_val.strip().upper()
    if not ticker_upper:
        st.error("Specificare un ticker valido.")
        st.stop()

    end_dt = date.today()
    try:
        start_dt = end_dt.replace(year=end_dt.year - years_back_val)
    except ValueError:
        start_dt = end_dt.replace(year=end_dt.year - years_back_val, day=max(1, end_dt.day - 1))
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # ── Init report HTML ──
    rp = [
        '<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">',
        f'<title>Report Stagionalità – {ticker_upper}</title>',
        REPORT_HTML_CSS,
        '</head><body>',
        f'<h1>Report Stagionalità: {ticker_upper}</h1>',
        f"<p style='text-align:center; font-size:small;'><i>Generato il {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"da KriterionQuant Seasonal Pattern Finder</i></p>",
    ]
    params_txt = (
        f"Ticker: {ticker_upper}\n"
        f"Periodo: {start_str} → {end_str} ({years_back_val} anni)\n"
        f"Scanner: Durata={min_len_val}-{max_len_val}gg, WinRate≥{min_win_rate_val}%, TopN={top_n_val}\n"
        f"Robustezza: Offset {range_robust_val[0]}–{range_robust_val[1]}gg"
    )
    rp.append("<div class='report-section'><h2>Parametri</h2>")
    rp.append(f"<pre>{params_txt}</pre></div>")

    # ── Download ──
    with st.spinner("📥 Download dati EODHD..."):
        hist_df = download_data(ticker_upper, start_str, end_str, get_ohlc=False)
    if hist_df is None or hist_df.empty:
        st.error("Download fallito. Verifica ticker e API key.")
        st.stop()

    # ── Pivot ──
    with st.spinner("📊 Calcolo Pivot Table..."):
        pivot_table = calculate_pivot_table(hist_df)
    if pivot_table is None or pivot_table.empty:
        st.error("Pivot Table non generabile.")
        st.stop()

    # ── Pattern Scan ──
    with st.spinner("🔍 Scansione pattern stagionali..."):
        all_found = find_seasonal_patterns(
            historical_data_df=hist_df, pivot=pivot_table,
            min_len=min_len_val, max_len=max_len_val,
            min_win_rate=min_win_rate_val, years_back_value=years_back_val,
            top_n_for_print=top_n_val, min_sharpe=0.0,
        )
    if all_found.empty:
        st.warning("Nessun pattern trovato con i criteri specificati.")
        st.stop()

    top_df = all_found.head(top_n_val).copy()

    # ── Persist ──
    st.session_state.historical_data_df = hist_df
    st.session_state.pivot_table = pivot_table
    st.session_state.top_df = top_df
    st.session_state.ui_params = {
        'ticker': ticker_upper, 'years_back': years_back_val,
        'min_len': min_len_val, 'max_len': max_len_val,
        'min_win_rate': min_win_rate_val, 'top_n': top_n_val,
        'range_robust': range_robust_val,
        'show_metrics_plots': show_metrics_plots, 'show_robustness': show_robustness,
        'show_annual_returns': show_annual_returns, 'show_detailed_charts': show_detailed_charts,
        'data_period_start': start_str, 'data_period_end': end_str,
    }
    st.session_state.report_html_parts = rp
    st.session_state.analysis_done = True
    st.session_state.selection_confirmed = False
    st.session_state.backtest_done = False


# ═════════════════════════════════════════════════════════════
# MAIN AREA – Tabs
# ═════════════════════════════════════════════════════════════
if st.session_state.analysis_done:
    top_df = st.session_state.top_df
    hist_df = st.session_state.historical_data_df
    pivot_table = st.session_state.pivot_table
    params = st.session_state.ui_params
    rp = st.session_state.report_html_parts
    tk = params['ticker']

    tab_pattern, tab_charts, tab_backtest = st.tabs([
        "📋 Pattern & Selezione",
        "📊 Grafici & Analisi",
        "🧪 Backtest Ensemble",
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 – Pattern & Selezione
    # ─────────────────────────────────────────────────────────
    with tab_pattern:
        st.subheader(f"Risultati per {tk}")

        # ── KPI Cards ──
        best = top_df.iloc[0] if not top_df.empty else None
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Pattern Trovati", f"{len(top_df)}")
        with k2:
            st.metric("Miglior Score", f"{best['CompositeScore']:.3f}" if best is not None else "–")
        with k3:
            st.metric("Miglior Win Rate", f"{best['WinRate']:.1f}%" if best is not None else "–")
        with k4:
            st.metric("Miglior Sharpe", f"{best['SharpeRatio']:.2f}" if best is not None and pd.notna(best['SharpeRatio']) else "–")
        with k5:
            avg_med = top_df['MedianReturn'].mean() * 100
            st.metric("Rend.Mediano Avg", f"{avg_med:+.2f}%")

        st.markdown("")

        # ── Curva Stagionale ──
        fig_main = plot_seasonal_pattern(hist_df, tk, pivot=pivot_table, patterns_df=top_df, top_n=params['top_n'])
        if fig_main:
            st.pyplot(fig_main, clear_figure=True)
            rp.append("<div class='report-section'><h2>Analisi Pattern Stagionali</h2>")
            rp.append("<div class='grid-item-full-width'>")
            rp.append(f"<h3>Stagionalità Media – {tk}</h3>")
            rp.append(fig_to_base64_html(fig_main, close_fig=False))
            rp.append("</div>")

        # ── Tabella Pattern ──
        st.markdown("#### Top Pattern Identificati")
        cols_disp = [
            "StartStr", "EndStr", "LenDays", "Direction", "AvgReturn", "MedianReturn",
            "WinRate", "Volatility", "SharpeRatio", "CompositeScore", "ProfitFactor", "NYears",
        ]
        act_cols = [c for c in cols_disp if c in top_df.columns]
        disp = top_df[act_cols].copy()
        rn = {
            "StartStr": "Inizio", "EndStr": "Fine", "LenDays": "Durata", "Direction": "Dir.",
            "AvgReturn": "Rend.Medio", "MedianReturn": "Rend.Mediano", "WinRate": "Win Rate",
            "Volatility": "Volat.", "SharpeRatio": "Sharpe", "CompositeScore": "Score",
            "ProfitFactor": "Profit F.", "NYears": "Anni",
        }
        disp.rename(columns=rn, inplace=True)
        fmt = {"Rend.Medio": "{:.2%}", "Rend.Mediano": "{:.2%}", "Win Rate": "{:.1f}%",
               "Volat.": "{:.2%}", "Sharpe": "{:.2f}", "Score": "{:.3f}", "Profit F.": "{:.2f}"}
        vf = {k: v for k, v in fmt.items() if k in disp.columns}
        styled_html = (
            disp.style.format(vf, na_rep='-')
            .background_gradient(subset=[c for c in ['Win Rate', 'Score', 'Profit F.'] if c in disp.columns],
                                 cmap='Greens', low=0.1)
            .to_html()
        )
        st.markdown(styled_html, unsafe_allow_html=True)
        rp.append("<div class='grid-item-full-width'>")
        rp.append(f"<h3>Top {len(disp)} Pattern</h3>")
        rp.append(styled_html)
        rp.append("</div>")

        # ── Data Editor per Selezione ──
        st.markdown("---")
        st.markdown("#### 🎯 Selezione Pattern e Pesi per Backtest")
        st.caption("Seleziona i pattern, assegna i pesi (≥ 0.0) e conferma.")

        ed_cols = ['StartStr', 'EndStr', 'LenDays', 'Direction', 'WinRate',
                   'MedianReturn', 'SharpeRatio', 'CompositeScore', 'NYears']
        ae = [c for c in ed_cols if c in top_df.columns]
        edf = top_df[ae].copy()
        edf.insert(0, 'Seleziona', False)
        nd = len(edf)
        dw = round(1.0 / nd, 2) if 0 < nd <= 5 else 0.10
        edf['PortfolioWeight'] = dw
        edf = edf.reset_index(drop=True)

        edited = st.data_editor(
            edf,
            column_config={
                "Seleziona": st.column_config.CheckboxColumn("✅", default=False),
                "PortfolioWeight": st.column_config.NumberColumn("Peso", min_value=0.0, max_value=10.0, step=0.01, format="%.2f"),
                "WinRate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                "MedianReturn": st.column_config.NumberColumn("Rend.Med.", format="%.2%%"),
                "SharpeRatio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "CompositeScore": st.column_config.NumberColumn("Score", format="%.3f"),
            },
            use_container_width=True, hide_index=True, num_rows="fixed",
            key="pattern_editor",
        )

        sel_mask = edited['Seleziona'] == True
        sel_n = sel_mask.sum()
        sw = edited.loc[sel_mask, 'PortfolioWeight'].sum()
        c_info_l, c_info_r = st.columns(2)
        with c_info_l:
            st.metric("Pattern Selezionati", f"{sel_n}")
        with c_info_r:
            st.metric("Somma Pesi", f"{sw:.2f}")

        if st.button("✅ Conferma Selezione e Pesi", type="primary"):
            if sel_n == 0:
                st.warning("Seleziona almeno un pattern.")
            else:
                idx = edited[sel_mask].index.tolist()
                sel_full = top_df.iloc[idx].copy()
                sel_full['PortfolioWeight'] = edited.loc[sel_mask, 'PortfolioWeight'].values
                st.session_state.selected_weighted_patterns_df = sel_full
                st.session_state.selection_confirmed = True
                st.success(f"✅ {sel_n} pattern confermati (pesi: {sw:.2f}).")

    # ─────────────────────────────────────────────────────────
    # TAB 2 – Grafici & Analisi
    # ─────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader(f"Visualizzazioni – {tk}")

        # ── Metriche Chiave ──
        if params.get('show_metrics_plots') and not top_df.empty:
            with st.expander("📊 Metriche Chiave dei Pattern", expanded=True):
                m_keys = ["AvgReturn", "MedianReturn", "SharpeRatio", "CompositeScore", "ProfitFactor"]
                m_names = ["Rend.Medio", "Rend.Mediano", "Sharpe", "Score", "Profit F."]
                labels = top_df['StartStr'] + " → " + top_df['EndStr']
                n_plots = sum(1 for k in m_keys if k in top_df.columns)
                if n_plots > 0:
                    fig_bar, axes = plt.subplots(
                        (n_plots + 1) // 2, 2,
                        figsize=(16, 5 * ((n_plots + 1) // 2)), squeeze=False,
                    )
                    axes = axes.flatten()
                    pi = 0
                    for mk, mn in zip(m_keys, m_names):
                        if mk not in top_df.columns:
                            continue
                        ax = axes[pi]
                        vals = top_df[mk].copy()
                        if mn in ["Rend.Medio", "Rend.Mediano"]:
                            vals *= 100
                            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
                        bar_colors = [DARK["long_green"] if v >= 0 else DARK["short_red"] for v in vals] if mn in ["Rend.Medio", "Rend.Mediano"] else [DARK["accent_blue"]] * len(vals)
                        ax.bar(labels.values, vals.values, color=bar_colors, alpha=0.8, edgecolor='none', width=0.65)
                        ax.set_title(mn, fontsize=11, fontweight='bold', color=DARK["text_bright"])
                        plt.setp(ax.get_xticklabels(), rotation=50, ha="right", fontsize=7, color=DARK["text_muted"])
                        ax.grid(True, axis='y', linestyle='--', alpha=0.3, color=DARK["grid"])
                        for spine in ax.spines.values():
                            spine.set_color(DARK["grid"])
                        pi += 1
                    for k in range(pi, len(axes)):
                        fig_bar.delaxes(axes[k])
                    fig_bar.suptitle(f"Metriche Chiave – Top {len(top_df)} Pattern", fontsize=14,
                                     fontweight='bold', color=DARK["text_bright"], y=0.99)
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    st.pyplot(fig_bar, clear_figure=True)
                    rp.append("<div class='grid-item-full-width'>")
                    rp.append(fig_to_base64_html(fig_bar, close_fig=False))
                    rp.append("</div>")

        # ── Rendimenti Annuali ──
        if params.get('show_annual_returns') and 'Returns' in top_df.columns:
            with st.expander("📅 Rendimenti Annuali per Pattern", expanded=False):
                for _, row_ar in top_df.iterrows():
                    ret = row_ar.get('Returns')
                    if isinstance(ret, (pd.Series, np.ndarray)) and len(ret) > 0:
                        fig_a = plot_yearly_returns_barchart(pd.DataFrame([row_ar]), top_n=1)
                        if fig_a:
                            st.pyplot(fig_a, clear_figure=True)
                            rp.append("<div class='grid-item'>")
                            rp.append(fig_to_base64_html(fig_a, close_fig=False))
                            rp.append("</div>")

        # ── Heatmap Robustezza ──
        if params.get('show_robustness') and not top_df.empty:
            with st.expander("🛡️ Heatmap Robustezza", expanded=True):
                min_off, max_off = params['range_robust']
                offsets = list(range(min_off, max_off + 1))
                pids = []
                seen = set()
                for _, r in top_df.iterrows():
                    pid = f"{r.get('StartStr','S')}_{r.get('EndStr','E')}"
                    pc = pid
                    cnt = 0
                    while pc in seen:
                        cnt += 1
                        pc = f"{pid}_{cnt}"
                    seen.add(pc)
                    pids.append(pc)

                rob = pd.DataFrame(index=offsets, columns=pids, dtype=float)
                rob.index.name = "Offset (±gg)"
                for pi2, (_, pr2) in enumerate(top_df.iterrows()):
                    sd2, ed2 = int(pr2['StartDay']), int(pr2['EndDay'])
                    cn2 = pids[pi2]
                    for off in offsets:
                        rob.loc[off, cn2] = get_shifted_window_avg_return(
                            pivot_table, sd2, ed2, off, params['years_back'],
                        )

                if not rob.empty and not rob.isna().all().all():
                    fig_r, ax_r = plt.subplots(
                        figsize=(max(12, len(pids) * 1.4), max(6, len(offsets) * 0.75))
                    )
                    sns.heatmap(
                        rob.dropna(axis=1, how='all'), cmap="RdYlGn", center=0,
                        annot=True, fmt=".2%",
                        annot_kws={"size": 9 if len(pids) <= 7 else 7, "color": DARK["text"]},
                        linewidths=0.5, linecolor=DARK["grid"],
                        cbar_kws={"label": "Rend. Medio Shiftato", "shrink": 0.8},
                        ax=ax_r,
                    )
                    cbar2 = ax_r.collections[0].colorbar
                    cbar2.ax.yaxis.label.set_color(DARK["text_muted"])
                    cbar2.ax.tick_params(colors=DARK["text_muted"])
                    ax_r.set_title(f"Robustezza Pattern – {tk}", fontsize=13,
                                   fontweight='bold', color=DARK["text_bright"], pad=12)
                    ax_r.set_xlabel("Pattern", fontsize=10, color=DARK["text_muted"])
                    ax_r.set_ylabel("Offset (gg)", fontsize=10, color=DARK["text_muted"])
                    plt.setp(ax_r.get_xticklabels(), rotation=55, ha="right", fontsize=8, color=DARK["text_muted"])
                    plt.setp(ax_r.get_yticklabels(), fontsize=9, color=DARK["text_muted"])
                    fig_r.tight_layout()
                    st.pyplot(fig_r, clear_figure=True)
                    rp.append("<div class='grid-item-full-width'>")
                    rp.append(fig_to_base64_html(fig_r, close_fig=False))
                    rp.append("</div>")

        # ── Grafici Dettagliati Asset ──
        if params.get('show_detailed_charts'):
            with st.expander("🔬 Visualizzazioni Dettagliate Asset", expanded=False):
                rp.append("<div class='report-section'><h2>Grafici Dettagliati</h2><div class='grid-container'>")
                detail_plots = [
                    (plot_calendar_heatmap, "Calendar Heatmap", (hist_df, tk), {}),
                    (plot_polar_seasonality, "Polare", (pivot_table, tk), {}),
                    (plot_monthly_box, "Box Plot Mensili", (hist_df, tk), {}),
                    (plot_stacked_patterns, "Stacked Patterns", (top_df, pivot_table), {'top_n': params['top_n']}),
                    (plot_yearly_overlay, "Overlay Annuale", (hist_df, tk), {}),
                    (plot_radar_monthly, "Radar Mensile", (hist_df, tk), {}),
                ]
                grid = st.columns(2)
                for i, (fn, title, args, kw) in enumerate(detail_plots):
                    try:
                        fig_d = fn(*args, **kw)
                        if fig_d:
                            with grid[i % 2]:
                                st.pyplot(fig_d, clear_figure=True)
                            rp.append("<div class='grid-item'>")
                            rp.append(f"<h3>{title}</h3>")
                            rp.append(fig_to_base64_html(fig_d, close_fig=False))
                            rp.append("</div>")
                    except Exception as e:
                        st.warning(f"Errore {title}: {e}")
                rp.append("</div></div>")

        rp.append("</div>")  # chiusura report-section
        st.session_state.report_html_parts = rp

    # ─────────────────────────────────────────────────────────
    # TAB 3 – Backtest Ensemble
    # ─────────────────────────────────────────────────────────
    with tab_backtest:
        st.subheader("Backtest Portafoglio Pattern")

        if not st.session_state.selection_confirmed:
            st.info("👈 Vai al tab **Pattern & Selezione**, seleziona i pattern e conferma per procedere.")
        else:
            sel_pats = st.session_state.selected_weighted_patterns_df
            rp = st.session_state.report_html_parts

            # ── KPI selezione ──
            kk1, kk2, kk3 = st.columns(3)
            with kk1:
                st.metric("Pattern nel Portafoglio", f"{len(sel_pats)}")
            with kk2:
                st.metric("Somma Pesi", f"{sel_pats['PortfolioWeight'].sum():.2f}")
            with kk3:
                avg_sc = sel_pats['CompositeScore'].mean()
                st.metric("Score Medio", f"{avg_sc:.3f}" if pd.notna(avg_sc) else "–")

            capital_base = st.number_input(
                "💰 Capitale Base per Trade (peso 1.0): $",
                min_value=100.0, value=10000.0, step=500.0,
            )

            if st.button("📈 Esegui Backtest e Genera Report", type="primary", use_container_width=True):
                data_start = pd.to_datetime(params['data_period_start'])
                first_yr = data_start.year
                if data_start.month > 1 or data_start.day > 2:
                    first_yr += 1
                last_yr = datetime.now().year - 1

                rp.append("<div class='report-section'><h2>Risultati Backtest</h2>")

                # Tabella composizione
                cb = ['StartStr', 'EndStr', 'Direction', 'CompositeScore', 'PortfolioWeight']
                db = sel_pats[[c for c in cb if c in sel_pats.columns]].copy()
                db.rename(columns={"StartStr": "Inizio", "EndStr": "Fine", "Direction": "Dir.",
                                   "CompositeScore": "Score", "PortfolioWeight": "Peso"}, inplace=True)
                bh = db.style.format({'Score': '{:.3f}', 'Peso': '{:.2f}'}, na_rep='-').set_caption(
                    f"Portafoglio – {tk}").to_html()
                st.markdown(bh, unsafe_allow_html=True)
                rp.append(bh)

                with st.spinner("⏳ Backtest in corso..."):
                    bkt_trades, bkt_ann, bkt_eq, _ = run_ensemble_backtest_single_ticker(
                        selected_patterns=sel_pats, ticker_ohlc_data=hist_df,
                        first_backtest_year=first_yr, last_backtest_year=last_yr,
                        capital_per_full_weight_trade=capital_base,
                        ticker_symbol_for_log=tk,
                    )

                if bkt_trades.empty:
                    st.warning("Nessun trade valido nel backtest.")
                    rp.append("<p>Nessun trade valido.</p>")
                else:
                    st.success(f"✅ Backtest completato: **{len(bkt_trades)}** trade eseguiti.")

                    # ── KPI Backtest ──
                    total_pnl = bkt_trades['PnL_Trade'].sum()
                    win_trades = (bkt_trades['PnL_Trade'] > 0).sum()
                    wr_bkt = win_trades / len(bkt_trades) * 100 if len(bkt_trades) > 0 else 0
                    max_dd = bkt_eq['CumulativePnL'].cummax().subtract(bkt_eq['CumulativePnL']).max() if not bkt_eq.empty else 0

                    bk1, bk2, bk3, bk4 = st.columns(4)
                    with bk1:
                        st.metric("P&L Totale", f"${total_pnl:,.2f}",
                                  delta=f"{'📈' if total_pnl > 0 else '📉'}")
                    with bk2:
                        st.metric("Trade Totali", f"{len(bkt_trades)}")
                    with bk3:
                        st.metric("Win Rate BKT", f"{wr_bkt:.1f}%")
                    with bk4:
                        st.metric("Max Drawdown", f"${max_dd:,.2f}")

                    # ── Plotly Equity Interattivo ──
                    fig_plotly = plot_ensemble_equity_plotly(
                        bkt_eq, f"Portafoglio su {tk}", selected_patterns_df=sel_pats,
                    )
                    if fig_plotly:
                        st.plotly_chart(fig_plotly, use_container_width=True)

                    # ── Matplotlib per HTML report ──
                    fig_mpl = plot_ensemble_equity_and_drawdown(
                        bkt_eq, f"Portafoglio su {tk}", selected_patterns_df=sel_pats,
                    )
                    if fig_mpl:
                        rp.append("<h3>Equity & Drawdown</h3>")
                        rp.append(fig_to_base64_html(fig_mpl, close_fig=True))

                    # ── P&L Annuale ──
                    if not bkt_ann.empty:
                        st.markdown("#### Riepilogo P&L Annuale")
                        ann_h = (
                            bkt_ann.style
                            .format({'PnL': '${:,.2f}', 'NumTrades': '{:d}'})
                            .bar(subset=['PnL'], align='zero', color=['#da3633', '#238636'])
                            .set_caption("P&L Annuale")
                            .to_html()
                        )
                        st.markdown(ann_h, unsafe_allow_html=True)
                        rp.append("<h3>P&L Annuale</h3>")
                        rp.append(ann_h)

                rp.append("</div></body></html>")
                st.session_state.report_html_parts = rp
                st.session_state.backtest_done = True

            # ── Download ──
            if st.session_state.backtest_done:
                st.markdown("---")
                final_html = "".join(st.session_state.report_html_parts)
                safe_tk = "".join(c if c.isalnum() else "_" for c in params['ticker'])
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="📥 Scarica Report HTML Completo",
                    data=final_html,
                    file_name=f"Report_{safe_tk}_{ts}.html",
                    mime="text/html",
                    use_container_width=True,
                )
