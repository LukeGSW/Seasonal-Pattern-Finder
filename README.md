KriterionQuant – Seasonal Pattern Finder
Applicazione web Streamlit per l'identificazione, la valutazione e il backtest di pattern stagionali su strumenti finanziari tramite dati EODHD.
Funzionalità
Scansione Pattern Stagionali: ricerca esaustiva di finestre temporali con rendimenti storicamente significativi
Composite Score: ranking basato su WinRate (20%), MedianReturn (20%) e SharpeRatio (60%) normalizzati a percentili
Heatmap di Robustezza: stress-test dei pattern con offset temporali variabili
Visualizzazioni: curva stagionale, calendar heatmap, radar mensile, box plot, overlay annuale, stacked patterns
Selezione & Pesatura: interfaccia interattiva (`st.data_editor`) per comporre il portafoglio di pattern
Backtest Ensemble: simulazione storica del portafoglio con equity curve, drawdown e P&L annuale
Report HTML: esportazione completa con grafici base64 embedded
Struttura Repository
```
kriterion-seasonal-finder/
├── app.py                  # Entry point Streamlit
├── requirements.txt        # Dipendenze
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # API key locale (NON committato)
└── core/
    ├── __init__.py
    ├── config.py            # Costanti globali e CSS report
    ├── data_fetcher.py      # Download dati EODHD
    ├── calculations.py      # Pivot table, day-of-year, robustezza
    ├── pattern_scanner.py   # Core engine di scansione pattern
    ├── visualizer.py        # Funzioni matplotlib/seaborn
    └── backtester.py        # Motore backtest ensemble
```
Setup Locale
```bash
# Clona il repo
git clone https://github.com/<tuo-user>/kriterion-seasonal-finder.git
cd kriterion-seasonal-finder

# Installa dipendenze
pip install -r requirements.txt

# Configura API key
echo 'EODHD_API_KEY = "la-tua-chiave"' > .streamlit/secrets.toml

# Avvia
streamlit run app.py
```
Deploy su Streamlit Community Cloud
Push del repo su GitHub (assicurati che `.streamlit/secrets.toml` sia nel `.gitignore`)
Vai su share.streamlit.io
Collega il repo e seleziona `app.py` come entry point
In Settings → Secrets, aggiungi:
```toml
   EODHD_API_KEY = "la-tua-chiave-EODHD"
   ```
Deploy
Autore
Luca De Cesare – KriterionQuant
