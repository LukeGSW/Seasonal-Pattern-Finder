# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
config.py – Variabili globali e costanti di configurazione.
"""

# Tasso risk-free annuale per il calcolo dello Sharpe Ratio
RISK_FREE_RATE_ANNUAL = 0.020

# Capitale di default per il backtest del portafoglio pattern
BACKTEST_CAPITAL_PER_FULL_WEIGHT_TRADE = 10000.0

# CSS per il report HTML esportabile
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
