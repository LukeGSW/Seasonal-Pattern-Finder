# -*- coding: utf-8 -*-
"""
KriterionQuant Seasonal Pattern Finder
data_fetcher.py – Funzioni di acquisizione dati da EODHD API.
Sostituisce google.colab.userdata con st.secrets per Streamlit.
"""

import pandas as pd
import requests
import streamlit as st
import traceback


def download_data(ticker: str, start: str, end: str, get_ohlc: bool = False) -> pd.DataFrame:
    """
    Scarica dati storici da EODHD.
    Se get_ohlc=True, tenta di includere open, high, low, close, volume.
    Altrimenti, si concentra su 'adjusted_close' (rinominato 'Adj Close').
    """
    st.write(f"[*] Download dati EODHD per {ticker} ({start}→{end}). Richiesto OHLC: {get_ohlc}")

    try:
        api_key = st.secrets["EODHD_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("ERRORE FATALE: API Key 'EODHD_API_KEY' non trovata in st.secrets.")
        return pd.DataFrame()

    api_url = f"https://eodhistoricaldata.com/api/eod/{ticker}"
    params = {
        'api_token': api_key,
        'from': start,
        'to': end,
        'period': 'd',
        'fmt': 'json',
        'order': 'a',
    }
    df_result = pd.DataFrame()

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        json_data = response.json()
        if not json_data or not isinstance(json_data, list) or not json_data:
            return df_result

        df_temp = pd.DataFrame(json_data)
        if df_temp.empty or 'date' not in df_temp.columns:
            return df_result

        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp.set_index('date', inplace=True)

        final_cols_to_keep = {}

        # Gestione Adjusted Close
        if 'adjusted_close' in df_temp.columns:
            final_cols_to_keep['adjusted_close'] = 'Adj Close'
        elif 'close' in df_temp.columns:
            final_cols_to_keep['close'] = 'Adj Close'

        if get_ohlc:
            ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlc_cols:
                if col in df_temp.columns:
                    if col == 'close' and 'adjusted_close' in df_temp.columns:
                        final_cols_to_keep[col] = 'close_unadj'
                    elif col == 'close' and 'adjusted_close' not in df_temp.columns:
                        pass
                    else:
                        final_cols_to_keep[col] = col

        if not final_cols_to_keep:
            st.warning(f"Nessuna colonna target trovata per {ticker}.")
            return pd.DataFrame()

        df_result = df_temp[list(final_cols_to_keep.keys())].copy()
        df_result.rename(columns=final_cols_to_keep, inplace=True)

        for col_name in df_result.columns:
            df_result[col_name] = pd.to_numeric(df_result[col_name], errors='coerce')
            if col_name == 'Adj Close':
                df_result[col_name] = df_result[col_name].ffill().bfill()
                df_result.dropna(subset=[col_name], inplace=True)

        if df_result.empty and 'Adj Close' in final_cols_to_keep.values():
            st.warning(f"DataFrame vuoto per {ticker} dopo pulizia NaN su 'Adj Close'.")
            return pd.DataFrame()

        if get_ohlc:
            essential_ohlc = ['open', 'high', 'low']
            if not all(col in df_result.columns for col in essential_ohlc):
                st.warning(f"Colonne OHLC essenziali mancanti per {ticker}. Colonne: {df_result.columns.tolist()}")

    except requests.exceptions.HTTPError as he:
        st.error(f"ERRORE HTTP {ticker}: Status {he.response.status_code}.")
        df_result = pd.DataFrame()
    except requests.exceptions.RequestException as re_err:
        st.error(f"ERRORE Rete/Connessione {ticker}: {re_err}")
        df_result = pd.DataFrame()
    except ValueError as ve:
        st.error(f"ERRORE Elaborazione Dati {ticker}: {ve}")
        df_result = pd.DataFrame()
    except Exception as exc:
        st.error(f"Errore generico {ticker}: {exc}")
        traceback.print_exc()
        df_result = pd.DataFrame()

    if df_result.empty:
        st.info(f"Download per {ticker} non ha prodotto dati validi.")
    return df_result
