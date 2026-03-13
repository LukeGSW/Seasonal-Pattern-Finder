# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
config.py – Variabili globali, costanti di configurazione, tema grafico.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ═══════════════════════════════════════════════════════════════
# Costanti di business (INVARIATE)
# ═══════════════════════════════════════════════════════════════
RISK_FREE_RATE_ANNUAL = 0.020
BACKTEST_CAPITAL_PER_FULL_WEIGHT_TRADE = 10000.0

# ═══════════════════════════════════════════════════════════════
# Dark Theme – Palette colori (stile Bloomberg / TradingView)
# ═══════════════════════════════════════════════════════════════
DARK = {
    # Sfondo e superfici
    "bg":           "#0e1117",
    "bg_card":      "#161b22",
    "bg_panel":     "#1c2333",
    # Testo
    "text":         "#c9d1d9",
    "text_muted":   "#8b949e",
    "text_bright":  "#f0f6fc",
    # Griglia e bordi
    "grid":         "#30363d",
    "border":       "#21262d",
    # Accenti
    "accent_blue":  "#58a6ff",
    "accent_cyan":  "#39d2c0",
    "accent_gold":  "#e3b341",
    "accent_purple":"#bc8cff",
    # Segnali
    "long_green":   "#3fb950",
    "long_green_bg":"#238636",
    "short_red":    "#f85149",
    "short_red_bg": "#da3633",
    "warn_orange":  "#d29922",
    # Palette sequenziale per grafici multi-serie
    "palette": [
        "#58a6ff", "#3fb950", "#e3b341", "#bc8cff", "#39d2c0",
        "#f778ba", "#f0883e", "#a5d6ff", "#7ee787", "#d2a8ff",
    ],
}

# ═══════════════════════════════════════════════════════════════
# Matplotlib rcParams – applica il dark theme globalmente
# ═══════════════════════════════════════════════════════════════
def apply_dark_theme():
    """Configura matplotlib con il dark theme KriterionQuant."""
    dark_rc = {
        # Figure
        "figure.facecolor":     DARK["bg"],
        "figure.edgecolor":     DARK["bg"],
        "figure.dpi":           120,
        # Axes
        "axes.facecolor":       DARK["bg_panel"],
        "axes.edgecolor":       DARK["grid"],
        "axes.labelcolor":      DARK["text"],
        "axes.titlecolor":      DARK["text_bright"],
        "axes.titlesize":       12,
        "axes.titleweight":     "bold",
        "axes.labelsize":       10,
        "axes.grid":            True,
        "axes.grid.which":      "major",
        "axes.prop_cycle":      mpl.cycler(color=DARK["palette"]),
        # Grid
        "grid.color":           DARK["grid"],
        "grid.alpha":           0.4,
        "grid.linewidth":       0.5,
        "grid.linestyle":       "--",
        # Ticks
        "xtick.color":          DARK["text_muted"],
        "ytick.color":          DARK["text_muted"],
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        # Legend
        "legend.facecolor":     DARK["bg_card"],
        "legend.edgecolor":     DARK["grid"],
        "legend.fontsize":      8,
        "legend.labelcolor":    DARK["text"],
        # Fonts
        "font.family":          "sans-serif",
        "font.size":            9,
        "text.color":           DARK["text"],
        # Savefig
        "savefig.facecolor":    DARK["bg"],
        "savefig.edgecolor":    DARK["bg"],
        "savefig.bbox":         "tight",
        "savefig.dpi":          150,
    }
    mpl.rcParams.update(dark_rc)


# ═══════════════════════════════════════════════════════════════
# Plotly dark template (per equity/drawdown interattivi)
# ═══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=DARK["bg"],
    plot_bgcolor=DARK["bg_panel"],
    font=dict(family="Segoe UI, sans-serif", color=DARK["text"], size=11),
    title_font=dict(size=16, color=DARK["text_bright"]),
    legend=dict(bgcolor=DARK["bg_card"], bordercolor=DARK["grid"], borderwidth=1),
    xaxis=dict(gridcolor=DARK["grid"], zerolinecolor=DARK["grid"]),
    yaxis=dict(gridcolor=DARK["grid"], zerolinecolor=DARK["grid"]),
    margin=dict(l=60, r=30, t=60, b=50),
)


# ═══════════════════════════════════════════════════════════════
# CSS custom per Streamlit (KPI cards, sidebar, branding)
# ═══════════════════════════════════════════════════════════════
STREAMLIT_CUSTOM_CSS = """
<style>
    /* ── KPI Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* ── Tabs styling ── */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* ── Data Editor ── */
    [data-testid="stDataEditor"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    /* ── Sidebar header ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* ── Download button ── */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# CSS per il report HTML esportabile (INVARIATO)
# ═══════════════════════════════════════════════════════════════
REPORT_HTML_CSS = """
<style>
    body { font-family: Verdana, sans-serif; margin: 15px; background-color: #f4f6f8; color: #333; font-size: 10pt; }
    h1, h2, h3 { color: #2a3f5f; border-bottom: 1px solid #ccc; padding-bottom: 6px; }
    h1 { font-size: 24px; text-align: center; margin-bottom: 25px; color: #1a237e; }
    h2 { font-size: 20px; margin-top: 30px; color: #283593; }
    h3 { font-size: 17px; margin-top: 20px; margin-bottom: 10px; color: #3949ab; }
    table { border-collapse: collapse; width: auto; max-width: 100%; margin: 15px auto; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 9pt; overflow-x: auto; display: block; }
    th, td { border: 1px solid #c8d2e0; padding: 7px 9px; text-align: center; }
    th { background-color: #3f51b5; color: white; font-weight: bold; }
    tr:nth-child(even) { background-color: #e8eaf6; }
    img { max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .report-section { background-color: #fff; padding: 20px 25px; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    pre { background-color: #f5f5f5; padding: 15px; border: 1px solid #e0e0e0; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 9.5pt; font-family: 'Courier New', Courier, monospace; }
    .grid-container { display: flex; flex-wrap: wrap; gap: 25px; justify-content: space-around; margin-top: 20px; }
    .grid-item { flex: 1 1 calc(50% - 25px); box-sizing: border-box; min-width: 420px; border: 1px solid #e7e7e7; padding:15px; background-color:#fdfdfd; margin-bottom:25px; border-radius: 6px;}
    .grid-item-full-width { flex-basis: 100%; box-sizing: border-box; margin-bottom: 25px; border: 1px solid #e7e7e7; padding:15px; background-color:#fdfdfd; border-radius: 6px;}
    p.didactic { font-size: 0.98em; color: #333; line-height: 1.65; margin-top: 8px; margin-bottom:18px; font-style: italic; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #3f51b5; border-radius: 4px;}
</style>
"""
