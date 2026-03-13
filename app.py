# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder – Streamlit App
app.py – Entry point dell'applicazione.

Fasi:
  1. Sidebar: Input parametri → Avvia Analisi → salva top_df in session_state
  2. Visualizzazione risultati + st.data_editor per selezione/pesatura pattern
  3. Backtest ensemble + generazione report HTML scaricabile
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
import traceback
import warnings
import io
import base64

# ── Moduli core ──
from core.config import RISK_FREE_RATE_ANNUAL, BACKTEST_CAPITAL_PER_FULL_WEIGHT_TRADE, REPORT_HTML_CSS
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
)
from core.backtester import run_ensemble_backtest_single_ticker

# ── Impostazioni globali ──
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue`")
warnings.filterwarnings("ignore", message="'M' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
plt.style.use("default")

# ── Page Config ──
st.set_page_config(
    page_title="KriterionQuant – Seasonal Pattern Finder",
    page_icon="📈",
    layout="wide",
)

st.title("📈 KriterionQuant – Seasonal Pattern Finder")
st.caption("Analizzatore di pattern stagionali con backtest ensemble")

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
# FASE 1 – Sidebar: Input Parametri e Avvio Scansione
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parametri Analisi")

    ticker_val = st.text_input("Ticker (EODHD)", value="BNB-USD.CC", help="Es: AAPL.US, BTC-USD.CC")
    years_back_val = st.slider("Anni Storico", min_value=5, max_value=30, value=20, step=1)
    min_len_val = st.slider("Min Durata Pattern (gg)", min_value=5, max_value=90, value=20, step=1)
    max_len_val = st.slider("Max Durata Pattern (gg)", min_value=5, max_value=120, value=60, step=1)
    min_win_rate_val = st.number_input("Min Win Rate (%)", min_value=50.0, max_value=100.0, value=75.0, step=0.5)
    top_n_val = st.slider("Top N Pattern", min_value=1, max_value=15, value=15, step=1)
    range_robust_val = st.slider("Range Robustezza (±gg)", min_value=0, max_value=7, value=(0, 3), step=1)

    st.markdown("---")
    st.subheader("Opzioni Visualizzazione")
    show_metrics_plots = st.checkbox("Grafici Metriche Chiave Pattern", value=True)
    show_robustness = st.checkbox("Heatmap Robustezza Pattern", value=True)
    show_annual_returns = st.checkbox("Grafici Rend. Annuali Pattern", value=True)
    show_detailed_charts = st.checkbox("Grafici Dettagliati Asset", value=True)

    st.markdown("---")
    run_analysis = st.button("🚀 Avvia Analisi Stagionale", use_container_width=True, type="primary")


# ── Esecuzione Fase 1 ──
if run_analysis:
    # Reset stato
    st.session_state.analysis_done = False
    st.session_state.selection_confirmed = False
    st.session_state.backtest_done = False
    st.session_state.top_df = pd.DataFrame()
    st.session_state.selected_weighted_patterns_df = pd.DataFrame()
    st.session_state.report_html_parts = []

    ticker_val_upper = ticker_val.strip().upper()
    if not ticker_val_upper:
        st.error("Specificare un ticker valido.")
        st.stop()

    # Calcolo date
    end_date_dt = date.today()
    try:
        start_date_dt = end_date_dt.replace(year=end_date_dt.year - years_back_val)
    except ValueError:
        start_date_dt = end_date_dt.replace(year=end_date_dt.year - years_back_val, day=max(1, end_date_dt.day - 1))
    start_date_str = start_date_dt.strftime("%Y-%m-%d")
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    # Inizializzazione report HTML
    report_parts = [
        '<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">',
        f'<title>Report Stagionalità – {ticker_val_upper}</title>',
        REPORT_HTML_CSS,
        '</head><body>',
        f'<h1>Report di Analisi Stagionale per: {ticker_val_upper}</h1>',
        f"<p style='text-align:center; font-size:small;'><i>Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} da KriterionQuant Seasonal Pattern Finder</i></p>",
    ]

    params_text = (
        f"Ticker: {ticker_val_upper}\n"
        f"Periodo Dati: {start_date_str} → {end_date_str} ({years_back_val} anni)\n"
        f"Scanner: Durata={min_len_val}-{max_len_val}gg, WinRate≥{min_win_rate_val}%, "
        f"Anni Richiesti={years_back_val}, TopN={top_n_val}\n"
        f"Robustezza: Offset {range_robust_val[0]}–{range_robust_val[1]}gg"
    )
    report_parts.append("<div class='report-section'><h2>Parametri dell'Analisi</h2>")
    report_parts.append(f"<pre>{params_text}</pre></div>")

    with st.spinner("📥 Download dati EODHD..."):
        hist_df = download_data(ticker_val_upper, start_date_str, end_date_str, get_ohlc=False)
    if hist_df is None or hist_df.empty:
        st.error("Download dati fallito. Verifica il ticker e la API key.")
        st.stop()
    st.success(f"Dati scaricati: {len(hist_df)} righe.")

    with st.spinner("📊 Calcolo Pivot Table..."):
        pivot_table = calculate_pivot_table(hist_df)
    if pivot_table is None or pivot_table.empty:
        st.error("Calcolo Pivot Table fallito.")
        st.stop()

    with st.spinner("🔍 Ricerca Pattern Stagionali..."):
        all_found = find_seasonal_patterns(
            historical_data_df=hist_df,
            pivot=pivot_table,
            min_len=min_len_val,
            max_len=max_len_val,
            min_win_rate=min_win_rate_val,
            years_back_value=years_back_val,
            top_n_for_print=top_n_val,
            min_sharpe=0.0,
        )

    if all_found.empty:
        st.warning("Nessun pattern trovato con i criteri specificati.")
        st.stop()

    top_df = all_found.head(top_n_val).copy()

    # Salva in session_state
    st.session_state.historical_data_df = hist_df
    st.session_state.pivot_table = pivot_table
    st.session_state.top_df = top_df
    st.session_state.ui_params = {
        'ticker': ticker_val_upper,
        'years_back': years_back_val,
        'min_len': min_len_val,
        'max_len': max_len_val,
        'min_win_rate': min_win_rate_val,
        'top_n': top_n_val,
        'range_robust': range_robust_val,
        'show_metrics_plots': show_metrics_plots,
        'show_robustness': show_robustness,
        'show_annual_returns': show_annual_returns,
        'show_detailed_charts': show_detailed_charts,
        'data_period_start': start_date_str,
        'data_period_end': end_date_str,
    }
    st.session_state.report_html_parts = report_parts
    st.session_state.analysis_done = True
    st.session_state.selection_confirmed = False
    st.session_state.backtest_done = False


# ═════════════════════════════════════════════════════════════
# FASE 2 – Visualizzazione Risultati e Selezione Pattern
# ═════════════════════════════════════════════════════════════
if st.session_state.analysis_done:
    top_df = st.session_state.top_df
    hist_df = st.session_state.historical_data_df
    pivot_table = st.session_state.pivot_table
    params = st.session_state.ui_params
    report_parts = st.session_state.report_html_parts
    ticker_val_upper = params['ticker']

    st.header(f"📊 Risultati Analisi Stagionale – {ticker_val_upper}")

    # ── Curva Stagionale Principale ──
    fig_seasonal_main = plot_seasonal_pattern(
        hist_df, ticker_val_upper, pivot=pivot_table,
        patterns_df=top_df, top_n=params['top_n'],
    )
    if fig_seasonal_main:
        st.pyplot(fig_seasonal_main, clear_figure=True)
        report_parts.append("<div class='report-section'><h2>Analisi dei Pattern Stagionali</h2>")
        report_parts.append("<div class='grid-item-full-width'>")
        report_parts.append(f"<h3>Curva di Stagionalità Media – {ticker_val_upper}</h3>")
        report_parts.append(fig_to_base64_html(fig_seasonal_main, close_fig=False))
        report_parts.append("</div>")

    # ── Tabella Pattern ──
    st.subheader("📋 Top Pattern Identificati")
    cols_display = [
        "StartStr", "EndStr", "LenDays", "Direction", "AvgReturn", "MedianReturn",
        "WinRate", "Volatility", "SharpeRatio", "CompositeScore", "ProfitFactor", "NYears",
    ]
    actual_cols = [c for c in cols_display if c in top_df.columns]
    display_df = top_df[actual_cols].copy()
    rename_map = {
        "StartStr": "Inizio", "EndStr": "Fine", "LenDays": "Durata", "Direction": "Dir.",
        "AvgReturn": "Rend.Medio", "MedianReturn": "Rend.Mediano", "WinRate": "Win Rate",
        "Volatility": "Volat.", "SharpeRatio": "Sharpe", "CompositeScore": "Score",
        "ProfitFactor": "Profit F.", "NYears": "Anni",
    }
    display_df.rename(columns=rename_map, inplace=True)
    fmt_map = {
        "Rend.Medio": "{:.2%}", "Rend.Mediano": "{:.2%}", "Win Rate": "{:.1f}%",
        "Volat.": "{:.2%}", "Sharpe": "{:.2f}", "Score": "{:.3f}", "Profit F.": "{:.2f}",
    }
    valid_fmt = {k: v for k, v in fmt_map.items() if k in display_df.columns}
    styled_html = (
        display_df.style
        .format(valid_fmt, na_rep='-')
        .background_gradient(
            subset=[c for c in ['Win Rate', 'Score', 'Profit F.'] if c in display_df.columns],
            cmap='Greens', low=0.1,
        )
        .to_html()
    )
    st.markdown(styled_html, unsafe_allow_html=True)
    report_parts.append("<div class='grid-item-full-width'>")
    report_parts.append(f"<h3>Top {len(display_df)} Pattern</h3>")
    report_parts.append(styled_html)
    report_parts.append("</div>")

    # ── Grafici Metriche Chiave ──
    if params.get('show_metrics_plots') and not top_df.empty:
        st.subheader("📊 Metriche Chiave dei Pattern")
        metrics_keys = ["AvgReturn", "MedianReturn", "SharpeRatio", "CompositeScore", "ProfitFactor"]
        metrics_names = ["Rend.Medio", "Rend.Mediano", "Sharpe", "Score", "Profit F."]
        labels = top_df['StartStr'] + "→" + top_df['EndStr']
        num_plots = sum(1 for k in metrics_keys if k in top_df.columns)
        if num_plots > 0:
            fig_bar, axes_bar = plt.subplots(
                (num_plots + 1) // 2, 2,
                figsize=(15, 5 * ((num_plots + 1) // 2)),
                squeeze=False,
            )
            axes_bar = axes_bar.flatten()
            plot_idx = 0
            for mk, mn in zip(metrics_keys, metrics_names):
                if mk not in top_df.columns:
                    continue
                ax = axes_bar[plot_idx]
                vals = top_df[mk].copy()
                if mn in ["Rend.Medio", "Rend.Mediano"]:
                    vals *= 100
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
                palette = "skyblue"
                if mn in ["Rend.Medio", "Rend.Mediano"]:
                    palette = ["mediumseagreen" if v >= 0 else "salmon" for v in vals]
                if isinstance(palette, list):
                    sns.barplot(x=labels.values, y=vals.values, ax=ax, palette=palette)
                else:
                    sns.barplot(x=labels.values, y=vals.values, ax=ax, color=palette)
                ax.set_title(mn, fontsize=11)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                plot_idx += 1
            for k in range(plot_idx, len(axes_bar)):
                fig_bar.delaxes(axes_bar[k])
            fig_bar.suptitle(f"Metriche Chiave – Top {len(top_df)} Pattern", fontsize=14, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            st.pyplot(fig_bar, clear_figure=True)
            report_parts.append("<div class='grid-item-full-width'>")
            report_parts.append(fig_to_base64_html(fig_bar, close_fig=False))
            report_parts.append("</div>")

    # ── Grafici Rendimenti Annuali ──
    if params.get('show_annual_returns') and 'Returns' in top_df.columns:
        st.subheader("📅 Rendimenti Annuali per Pattern")
        for _, row_ar in top_df.iterrows():
            ret = row_ar.get('Returns')
            if isinstance(ret, (pd.Series, np.ndarray)) and len(ret) > 0:
                fig_annual = plot_yearly_returns_barchart(pd.DataFrame([row_ar]), top_n=1)
                if fig_annual:
                    st.pyplot(fig_annual, clear_figure=True)
                    report_parts.append("<div class='grid-item'>")
                    report_parts.append(fig_to_base64_html(fig_annual, close_fig=False))
                    report_parts.append("</div>")

    # ── Heatmap Robustezza ──
    if params.get('show_robustness') and not top_df.empty:
        st.subheader("🛡️ Heatmap Robustezza Pattern")
        min_offset, max_offset = params['range_robust']
        offsets = list(range(min_offset, max_offset + 1))
        pattern_ids = []
        ids_seen = set()
        for _, r in top_df.iterrows():
            p_id = f"{r.get('StartStr','S')}_{r.get('EndStr','E')}"
            p_id_curr = p_id
            cnt = 0
            while p_id_curr in ids_seen:
                cnt += 1
                p_id_curr = f"{p_id}_{cnt}"
            ids_seen.add(p_id_curr)
            pattern_ids.append(p_id_curr)

        rob_matrix = pd.DataFrame(index=offsets, columns=pattern_ids, dtype=float)
        rob_matrix.index.name = "Offset (±gg)"
        for p_idx, (_, pr) in enumerate(top_df.iterrows()):
            sd = int(pr['StartDay'])
            ed = int(pr['EndDay'])
            col_name = pattern_ids[p_idx]
            for off in offsets:
                avg_ret = get_shifted_window_avg_return(
                    pivot_table, sd, ed, off, params['years_back'],
                )
                rob_matrix.loc[off, col_name] = avg_ret

        if not rob_matrix.empty and not rob_matrix.isna().all().all():
            fig_rob, ax_rob = plt.subplots(
                figsize=(max(11, len(pattern_ids) * 1.3), max(6, len(offsets) * 0.7))
            )
            sns.heatmap(
                rob_matrix.dropna(axis=1, how='all'), cmap="RdYlGn", center=0,
                annot=True, fmt=".2%", annot_kws={"size": 8 if len(pattern_ids) <= 7 else 7},
                linewidths=.5, linecolor='lightgray',
                cbar_kws={"label": "Rend. Medio Shiftato (%)"},
                ax=ax_rob,
            )
            ax_rob.set_title(f"Heatmap Robustezza – {ticker_val_upper}", fontsize=13, pad=12)
            ax_rob.set_xlabel("Pattern (Inizio_Fine)", fontsize=11)
            ax_rob.set_ylabel("Offset Applicato", fontsize=11)
            plt.setp(ax_rob.get_xticklabels(), rotation=60, ha="right", fontsize=9)
            fig_rob.tight_layout()
            st.pyplot(fig_rob, clear_figure=True)
            report_parts.append("<div class='grid-item-full-width'>")
            report_parts.append(fig_to_base64_html(fig_rob, close_fig=False))
            report_parts.append("</div>")

    # ── Grafici Dettagliati Asset ──
    if params.get('show_detailed_charts'):
        st.subheader("🔬 Visualizzazioni Dettagliate Asset")
        report_parts.append("<div class='report-section'><h2>Visualizzazioni Dettagliate</h2><div class='grid-container'>")
        detailed_plots = [
            (plot_calendar_heatmap, "Calendar Heatmap", (hist_df, ticker_val_upper), {}),
            (plot_polar_seasonality, "Grafico Polare", (pivot_table, ticker_val_upper), {}),
            (plot_monthly_box, "Box Plot Mensili", (hist_df, ticker_val_upper), {}),
            (plot_stacked_patterns, "Stacked Patterns", (top_df, pivot_table), {'top_n': params['top_n']}),
            (plot_yearly_overlay, "Overlay Annuale", (hist_df, ticker_val_upper), {}),
            (plot_radar_monthly, "Radar Mensile", (hist_df, ticker_val_upper), {}),
        ]
        cols_grid = st.columns(2)
        for i, (fn, title, args, kwargs) in enumerate(detailed_plots):
            try:
                fig_det = fn(*args, **kwargs)
                if fig_det:
                    with cols_grid[i % 2]:
                        st.pyplot(fig_det, clear_figure=True)
                    report_parts.append("<div class='grid-item'>")
                    report_parts.append(f"<h3>{title}</h3>")
                    report_parts.append(fig_to_base64_html(fig_det, close_fig=False))
                    report_parts.append("</div>")
            except Exception as e:
                st.warning(f"Errore generazione {title}: {e}")
        report_parts.append("</div></div>")

    report_parts.append("</div>")  # chiusura report-section principale
    st.session_state.report_html_parts = report_parts

    # ═════════════════════════════════════════════════════════════
    # SELEZIONE PATTERN con st.data_editor
    # ═════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("🎯 Selezione Pattern e Pesi per Backtest")
    st.caption("Seleziona i pattern da includere nel backtest e assegna i pesi (≥ 0.0).")

    # Prepara dataframe per l'editor
    editor_cols = ['StartStr', 'EndStr', 'LenDays', 'Direction', 'WinRate', 'MedianReturn',
                   'SharpeRatio', 'CompositeScore', 'NYears']
    actual_editor_cols = [c for c in editor_cols if c in top_df.columns]
    editor_df = top_df[actual_editor_cols].copy()
    editor_df.insert(0, 'Seleziona', False)
    num_displayed = len(editor_df)
    default_w = round(1.0 / num_displayed, 2) if 0 < num_displayed <= 5 else 0.10
    editor_df['PortfolioWeight'] = default_w
    editor_df = editor_df.reset_index(drop=True)

    edited_df = st.data_editor(
        editor_df,
        column_config={
            "Seleziona": st.column_config.CheckboxColumn("✅", default=False),
            "PortfolioWeight": st.column_config.NumberColumn("Peso", min_value=0.0, max_value=10.0, step=0.01, format="%.2f"),
            "WinRate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
            "MedianReturn": st.column_config.NumberColumn("Rend.Mediano", format="%.2%%"),
            "SharpeRatio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "CompositeScore": st.column_config.NumberColumn("Score", format="%.3f"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="pattern_editor",
    )

    selected_mask = edited_df['Seleziona'] == True
    selected_count = selected_mask.sum()
    sum_weights = edited_df.loc[selected_mask, 'PortfolioWeight'].sum()
    st.info(f"Pattern selezionati: **{selected_count}** — Somma pesi: **{sum_weights:.2f}**")

    confirm_selection = st.button("✅ Conferma Selezione e Pesi", type="primary")
    if confirm_selection:
        if selected_count == 0:
            st.warning("Seleziona almeno un pattern.")
        else:
            # Ricostruisci il df completo con tutte le colonne originali
            selected_indices = edited_df[selected_mask].index.tolist()
            selected_full = top_df.iloc[selected_indices].copy()
            selected_full['PortfolioWeight'] = edited_df.loc[selected_mask, 'PortfolioWeight'].values
            st.session_state.selected_weighted_patterns_df = selected_full
            st.session_state.selection_confirmed = True
            st.success(f"{selected_count} pattern confermati (somma pesi: {sum_weights:.2f}).")


# ═════════════════════════════════════════════════════════════
# FASE 3 – Backtest Ensemble
# ═════════════════════════════════════════════════════════════
if st.session_state.selection_confirmed:
    st.markdown("---")
    st.header("🧪 Backtest Portafoglio Pattern")

    selected_patterns = st.session_state.selected_weighted_patterns_df
    hist_df = st.session_state.historical_data_df
    params = st.session_state.ui_params
    report_parts = st.session_state.report_html_parts
    ticker_val_upper = params['ticker']

    capital_base = st.number_input(
        "Capitale Base per Trade (peso 1.0): $",
        min_value=100.0, value=10000.0, step=500.0,
    )

    run_backtest = st.button("📈 Esegui Backtest e Genera Report", type="primary")
    if run_backtest:
        # Calcola primo anno valido
        data_start = pd.to_datetime(params['data_period_start'])
        first_bkt_year = data_start.year
        if data_start.month > 1 or data_start.day > 2:
            first_bkt_year += 1
        last_bkt_year = datetime.now().year - 1

        report_parts.append("<div class='report-section'><h2>Risultati Backtest Portafoglio Pattern</h2>")

        # Tabella pattern nel backtest
        cols_bkt = ['StartStr', 'EndStr', 'Direction', 'CompositeScore', 'PortfolioWeight']
        df_bkt_display = selected_patterns[[c for c in cols_bkt if c in selected_patterns.columns]].copy()
        df_bkt_display.rename(columns={
            "StartStr": "Inizio", "EndStr": "Fine", "Direction": "Dir.",
            "CompositeScore": "Score", "PortfolioWeight": "Peso",
        }, inplace=True)
        bkt_fmt = {'Score': '{:.3f}', 'Peso': '{:.2f}'}
        bkt_html = df_bkt_display.style.format(bkt_fmt, na_rep='-').set_caption(
            f"Configurazione Portafoglio – {ticker_val_upper}"
        ).to_html()
        st.markdown(bkt_html, unsafe_allow_html=True)
        report_parts.append(bkt_html)

        with st.spinner("⏳ Esecuzione backtest..."):
            bkt_trades, bkt_annual_pnl, bkt_equity, _ = run_ensemble_backtest_single_ticker(
                selected_patterns=selected_patterns,
                ticker_ohlc_data=hist_df,
                first_backtest_year=first_bkt_year,
                last_backtest_year=last_bkt_year,
                capital_per_full_weight_trade=capital_base,
                ticker_symbol_for_log=ticker_val_upper,
            )

        if bkt_trades.empty:
            st.warning("Nessun trade valido eseguito durante il backtest.")
            report_parts.append("<p>Nessun trade valido eseguito.</p>")
        else:
            st.success(f"Backtest completato: {len(bkt_trades)} trade eseguiti.")

            # Equity + Drawdown
            fig_equity = plot_ensemble_equity_and_drawdown(
                bkt_equity,
                f"Portafoglio Pattern su {ticker_val_upper}",
                selected_patterns_df=selected_patterns,
            )
            if fig_equity:
                st.pyplot(fig_equity, clear_figure=True)
                report_parts.append("<h3>Equity Curve e Drawdown</h3>")
                report_parts.append(fig_to_base64_html(fig_equity, close_fig=False))

            # P&L Annuale
            if not bkt_annual_pnl.empty:
                st.subheader("📊 Riepilogo P&L Annuale")
                ann_fmt = {'PnL': '${:,.2f}', 'NumTrades': '{:d}'}
                ann_html = (
                    bkt_annual_pnl.style
                    .format(ann_fmt)
                    .bar(subset=['PnL'], align='zero', color=['lightcoral', 'mediumseagreen'])
                    .set_caption("P&L Annuale")
                    .to_html()
                )
                st.markdown(ann_html, unsafe_allow_html=True)
                report_parts.append("<h3>P&L Annuale</h3>")
                report_parts.append(ann_html)

        # Chiusura report HTML
        report_parts.append("</div></body></html>")
        st.session_state.report_html_parts = report_parts
        st.session_state.backtest_done = True

    # ── Download Report ──
    if st.session_state.backtest_done:
        st.markdown("---")
        final_html = "".join(st.session_state.report_html_parts)
        safe_ticker = "".join(c if c.isalnum() else "_" for c in params['ticker'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Report_{safe_ticker}_{timestamp}.html"
        st.download_button(
            label="📥 Scarica Report HTML Completo",
            data=final_html,
            file_name=filename,
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
