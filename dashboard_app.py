    # dashboard_app.py
from flask import Flask, jsonify, render_template, request
import pandas as pd
import pandas_ta as ta # Nécessaire pour recalculer les indicateurs
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
import logging
import os
import time
import csv
import requests
import hashlib
import hmac
import base64

# --- Configuration des Chemins ---
PERSISTENT_DATA_PATH = os.getenv("PERSISTENT_DATA_PATH", "/app/data") 
STATE_FILE = os.path.join(PERSISTENT_DATA_PATH, "paper_bot_state.json")
TRADES_LOG_FILE = os.path.join(PERSISTENT_DATA_PATH, "paper_bot_trades.csv")
ACTIVITY_LOG_FILE = os.path.join(PERSISTENT_DATA_PATH, "paper_bot_activity.log")
KLINES_BASE_DIR = os.path.join(PERSISTENT_DATA_PATH, "klines_data")
# Le paper_trader sauvegarde avec SYMBOL et TIMEFRAME_KUCOIN, donc on les définit ici aussi
SYMBOL_FOR_KLINES = "ETH-USDT" # Doit correspondre à celui du paper_trader
TIMEFRAME_FOR_KLINES = '1hour' # Doit correspondre à celui du paper_trader
LOCAL_KLINES_FILE = os.path.join(KLINES_BASE_DIR, f"local_klines_{SYMBOL_FOR_KLINES.replace('-', '_')}_{TIMEFRAME_FOR_KLINES}.csv")
HISTORICAL_DATA_FETCH_LIMIT = 350 # Utilisé par get_market_klines
TIMEFRAME_TO_MINUTES = {
    '1min': 1, '3min': 3, '5min': 5, '15min': 15, '30min': 30, 
    '1hour': 60, '2hour': 120 # Ajoutez d'autres si nécessaire pour KuCoin
}


# Paramètres de la stratégie (BEST_PARAMS de votre paper_trader.py)
BEST_PARAMS = {
    'adx_len': 20, 'adx_trend_th': 28, 'adx_range_th': 22,
    'sma_short': 30, 'sma_long': 50, 
    'rsi_len': 10, 'rsi_os': 35, 'rsi_ob_exit': 65,
    'sl': 0.02, 'tp': 0.088 
}
INITIAL_PAPER_CAPITAL = 44.42 # Doit correspondre à celui de votre paper_trader.py


BASE_URL = 'https://api.kucoin.com'
BEST_PARAMS = {
    'adx_len': 20, 'adx_trend_th': 28, 'adx_range_th': 22,
    'sma_short': 30, 'sma_long': 50, 
    'rsi_len': 10, 'rsi_os': 35, 'rsi_ob_exit': 65,
    'sl': 0.02, 'tp': 0.088 
}
STRATEGY_NAME = (f"PaperBot_ADX{BEST_PARAMS['adx_len']}"
                 f"({BEST_PARAMS['adx_trend_th']}/{BEST_PARAMS['adx_range_th']})_"
                 f"SMA({BEST_PARAMS['sma_short']}/{BEST_PARAMS['sma_long']})_"
                 f"RSI({BEST_PARAMS['rsi_len']}_{BEST_PARAMS['rsi_os']}/{BEST_PARAMS['rsi_ob_exit']})_"
                 f"SL{BEST_PARAMS['sl']:.1%}_TP{BEST_PARAMS['tp']:.1%}")

RISK_PER_TRADE_PCT = 0.02
COMMISSION_RATE = 0.001 # Taker fee 0.1%

STATE_FILE = "paper_bot_state.json"
TRADES_LOG_FILE = "paper_bot_trades.csv"
LOG_FILE_ACTIVITY = "paper_bot_activity.log" # Renommé pour clarté


app = Flask(__name__)

# --- Fonctions Utilitaires (copiées/adaptées de vos scripts précédents) ---

def save_klines_locally(df_new_klines, symbol, timeframe):
    # S'assurer que KLINES_BASE_DIR est accessible ou créé
    if not os.path.exists(KLINES_BASE_DIR):
        try:
            os.makedirs(KLINES_BASE_DIR, exist_ok=True)
            logging.info(f"Création du répertoire de klines: {KLINES_BASE_DIR}")
        except Exception as e:
            logging.error(f"Impossible de créer le répertoire klines {KLINES_BASE_DIR}: {e}")
            # Fallback ou erreur ? Pour l'instant, on logue.
            # Si on ne peut pas créer le dossier, la sauvegarde échouera.
            # On pourrait choisir de ne pas sauvegarder si le dossier n'est pas là.
            return 

    filename = os.path.join(KLINES_BASE_DIR, f"local_klines_{symbol.replace('-', '_')}_{timeframe}.csv")
    df_to_save = df_new_klines.copy()
    if df_to_save.index.name is None: df_to_save.index.name = 'Open time'
    
    file_exists_and_not_empty = os.path.exists(filename) and os.path.getsize(filename) > 0

    if file_exists_and_not_empty:
        try:
            df_existing = pd.read_csv(filename, index_col='Open time', parse_dates=True, sep=';')
            if not df_existing.empty:
                # Assurer la cohérence des fuseaux horaires avant la concaténation
                if df_existing.index.tz is None: df_existing.index = df_existing.index.tz_localize('UTC')
                else: df_existing.index = df_existing.index.tz_convert('UTC')
                
                if df_to_save.index.tz is None: df_to_save.index = df_to_save.index.tz_localize('UTC')
                else: df_to_save.index = df_to_save.index.tz_convert('UTC')

                df_combined = pd.concat([df_existing, df_to_save])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                df_combined.to_csv(filename, sep=';')
                logging.info(f"Données de marché (klines) mises à jour dans {filename} ({len(df_combined)} lignes).")
                return 
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du fichier klines local {filename}: {e}. Tentative d'écrasement.")
    
    # Si le fichier n'existe pas, est vide, ou si l'erreur de fusion, on l'écrase/crée.
    try:
        df_to_save.to_csv(filename, sep=';')
        logging.info(f"Données de marché (klines) sauvegardées dans {filename} ({len(df_to_save)} lignes).")
    except Exception as e:
        logging.error(f"Erreur finale de sauvegarde des klines dans {filename}: {e}")


def get_market_klines(symbol=SYMBOL_FOR_KLINES, timeframe_type=TIMEFRAME_FOR_KLINES, limit=HISTORICAL_DATA_FETCH_LIMIT):
    path = '/api/v1/market/candles'
    end_time_s = int(time.time())
    # S'assurer que timeframe_minutes est un nombre
    tf_minutes_for_calc = TIMEFRAME_TO_MINUTES.get(timeframe_type, 60) # Défaut à 60 minutes (1 heure)
    start_time_s = end_time_s - (limit * tf_minutes_for_calc * 60 * 1.2) # 1.2 pour marge de sécurité
    
    params = {'symbol': symbol, 'type': timeframe_type, 'startAt': int(start_time_s)}
    logging.info(f"Récupération de ~{limit} bougies pour {symbol} type {timeframe_type} depuis {datetime.fromtimestamp(start_time_s, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        response = requests.get(BASE_URL + path, params=params, timeout=20)
        if response.status_code == 200:
            data = response.json().get('data', [])
            if not data:
                logging.warning(f"Aucune donnée kline reçue pour {symbol} {timeframe_type}.")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['Open time', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover'])
            df['Open time'] = pd.to_datetime(df['Open time'], unit='s', utc=True)
            df.set_index('Open time', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            df.sort_index(ascending=True, inplace=True) 
            
            if len(df) >= limit * 0.85: # Accepter si on a au moins 85% des données attendues
                df_to_use = df.iloc[-limit:] if len(df) > limit else df 
                save_klines_locally(df_to_use, symbol, timeframe_type) 
                return df_to_use
            else:
                logging.warning(f"Pas assez de bougies ({len(df)}/{limit*0.85:.0f} min) pour {symbol} {timeframe_type}.")
                return pd.DataFrame()
        elif response.status_code == 429: # Too Many Requests
            logging.warning(f"Rate limit (429) pour {symbol} {timeframe_type}. Le dashboard ne récupérera pas de données cette fois.")
            return pd.DataFrame() # Ne pas relancer de boucle ici pour le dashboard
        else:
            logging.error(f"Erreur API KuCoin (klines): {response.status_code} - {response.text} pour GET {path} avec params {params}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de requête (klines) vers KuCoin: {e}")
        return pd.DataFrame()


def generate_sma_crossover_signals(df_period_in, sma_short_len, sma_long_len):
    # ... (Copiez la fonction de paper_trader.py)
    df_p = df_period_in.copy(); sma_short_col = f'SMA_{sma_short_len}'; sma_long_col = f'SMA_{sma_long_len}'
    if sma_short_col not in df_p.columns or df_p[sma_short_col].isnull().all(): df_p[sma_short_col] = df_p['Close'].rolling(window=sma_short_len, min_periods=sma_short_len).mean()
    if sma_long_col not in df_p.columns or df_p[sma_long_col].isnull().all(): df_p[sma_long_col] = df_p['Close'].rolling(window=sma_long_len, min_periods=sma_long_len).mean()
    df_p.dropna(subset=[sma_short_col, sma_long_col], inplace=True)
    if df_p.empty: return df_p.assign(signal_SmaCross=0)
    if not all(col in df_p.columns for col in [sma_short_col, sma_long_col]): return df_p.assign(signal_SmaCross=0)
    df_p['position_SmaCross'] = np.where(df_p[sma_short_col] > df_p[sma_long_col], 1, -1) 
    df_p['signal_SmaCross'] = 0; diff_values = df_p['position_SmaCross'].diff().fillna(0)
    df_p.loc[diff_values == 2, 'signal_SmaCross'] = 1; df_p.loc[diff_values == -2, 'signal_SmaCross'] = -1
    return df_p

def generate_simple_rsi_signals(df_period_in, rsi_length, rsi_oversold, rsi_overbought_exit):
    # ... (Copiez la fonction de paper_trader.py)
    df_p = df_period_in.copy(); rsi_col_name = f'RSI_{rsi_length}'
    if rsi_col_name not in df_p.columns or df_p[rsi_col_name].isnull().all():
        if hasattr(df_p, 'ta'): df_p.ta.rsi(length=rsi_length, append=True, col_names=(rsi_col_name,)); df_p.dropna(subset=[rsi_col_name], inplace=True)
        else: return df_p.assign(signal_SimpleRsi=0)
    if df_p.empty or rsi_col_name not in df_p.columns: return df_p.assign(signal_SimpleRsi=0)
    df_p['signal_SimpleRsi'] = 0
    df_p.loc[(df_p[rsi_col_name].shift(1).fillna(50) < rsi_oversold) & (df_p[rsi_col_name] >= rsi_oversold), 'signal_SimpleRsi'] = 1
    df_p.loc[(df_p[rsi_col_name].shift(1).fillna(50) > rsi_overbought_exit) & (df_p[rsi_col_name] <= rsi_overbought_exit), 'signal_SimpleRsi'] = -1
    return df_p

def calculate_performance_metrics(df_trades, initial_capital):
    if df_trades.empty:
        return {
            "total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": None, # Utiliser None pour JSON
            "num_trades": 0, "win_rate_pct": 0, "profit_factor": 0,
            "current_equity": initial_capital, "pnl_total_abs":0,
            "equity_curve_values": [initial_capital], 
            "equity_curve_timestamps": [datetime.now(timezone.utc).isoformat()] # Timestamp de début
        }

    current_equity = initial_capital
    equity_over_time_values = [initial_capital]
    
    # Utiliser le timestamp du premier trade s'il existe, sinon une valeur par défaut
    first_trade_timestamp = None
    if not df_trades.empty and 'EntryTimeUTC' in df_trades.columns and pd.notnull(df_trades['EntryTimeUTC'].iloc[0]):
        try:
            first_trade_timestamp = pd.to_datetime(df_trades['EntryTimeUTC'].iloc[0]).isoformat()
        except: # Si la conversion échoue
            first_trade_timestamp = datetime.now(timezone.utc).isoformat() 
    else:
        first_trade_timestamp = datetime.now(timezone.utc).isoformat() 
        
    equity_over_time_timestamps = [first_trade_timestamp]


    df_trades['PnL_Abs'] = pd.to_numeric(df_trades['PnL_Abs'], errors='coerce').fillna(0)
    # S'assurer que ExitTimeUTC est bien un objet datetime avant d'essayer .isoformat()
    df_trades['ExitTimeUTC_dt'] = pd.to_datetime(df_trades['ExitTimeUTC'], errors='coerce')


    for idx, row in df_trades.iterrows():
        current_equity += row['PnL_Abs']
        equity_over_time_values.append(current_equity)
        if pd.notnull(row['ExitTimeUTC_dt']):
            equity_over_time_timestamps.append(row['ExitTimeUTC_dt'].isoformat())
        else: # Fallback si la date de sortie est manquante
            equity_over_time_timestamps.append(datetime.now(timezone.utc).isoformat()) 
    
    # Convertir les timestamps de l'index en string pour JSON
    # equity_series = pd.Series(equity_over_time_values, index=pd.to_datetime(equity_over_time_timestamps, utc=True, errors='coerce'))
    # equity_series = equity_series[equity_series.index.notna()]
    # Plutôt que de créer une série pandas juste pour ça, on a déjà les listes
    
    total_return_pct = (current_equity / initial_capital - 1) if initial_capital > 0 else 0
    
    # Calcul du Max Drawdown sur les valeurs d'équité
    temp_equity_series_for_dd = pd.Series(equity_over_time_values)
    peak_dd = temp_equity_series_for_dd.expanding(min_periods=1).max()
    drawdown_dd = (temp_equity_series_for_dd / peak_dd) - 1
    max_drawdown_pct = drawdown_dd.min() if not drawdown_dd.empty else 0

    sharpe_ratio = None # Initialiser à None (valide en JSON)
    # Calcul du Sharpe Ratio (simplifié, basé sur les rendements des trades)
    if not df_trades.empty and 'PnL_Pct' in df_trades.columns:
        trade_returns = pd.to_numeric(df_trades['PnL_Pct'], errors='coerce').fillna(0)
        if len(trade_returns) > 1 and trade_returns.std(ddof=0) != 0:
            # Sharpe par trade (non annualisé)
            # sharpe_ratio_per_trade = trade_returns.mean() / trade_returns.std(ddof=0)
            # Pour une estimation annualisée, c'est plus complexe ici. Laisser à None pour l'instant.
             pass # Laisser sharpe_ratio à None ou implémenter une annualisation plus tard

    num_trades = len(df_trades)
    winning_trades = df_trades[df_trades['PnL_Abs'] > 0]
    win_rate_pct = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0
    
    total_profit = df_trades[df_trades['PnL_Abs'] > 0]['PnL_Abs'].sum()
    total_loss = abs(df_trades[df_trades['PnL_Abs'] < 0]['PnL_Abs'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    return {
        "total_return_pct": total_return_pct * 100,
        "max_drawdown_pct": max_drawdown_pct * 100,
        "sharpe_ratio": sharpe_ratio, # Sera None si non calculé
        "num_trades": num_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor if profit_factor != float('inf') else None, # Renvoyer None si infini
        "current_equity": current_equity,
        "pnl_total_abs": current_equity - initial_capital,
        "equity_curve_values": equity_over_time_values,
        "equity_curve_timestamps": equity_over_time_timestamps 
    }
@app.route('/')
def index_page():
    return render_template("index.html") 

@app.route('/dashboard_data') # Endpoint unique pour toutes les données du dashboard
def dashboard_data():
    state_data = {}
    trades_data = []
    performance_data = {}
    activity_data = []

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: state_data = json.load(f)
            if state_data.get("entry_time_utc_str"): # Convertir si c'est une string
                 state_data["entry_time_utc"] = state_data["entry_time_utc_str"] # Garder en string pour JSON
        except Exception as e: state_data = {"error": f"Erreur lecture état: {str(e)}"}
    else: state_data = {"error": "Fichier d'état non trouvé"}
    
    if os.path.exists(TRADES_LOG_FILE):
        try:
            df_trades = pd.read_csv(TRADES_LOG_FILE, delimiter=';')
            df_trades.fillna('', inplace=True) # Remplacer NaN par string vide pour JSON
            trades_data = df_trades.to_dict(orient='records')
            # Utiliser le capital initial du bot pour le calcul de performance
            initial_capital_for_calc = state_data.get("paper_capital_at_start_of_log", INITIAL_PAPER_CAPITAL) 
            # Si le fichier d'état contient le capital de départ du log, l'utiliser, sinon celui par défaut.
            # Il serait mieux que le paper_bot enregistre son INITIAL_PAPER_CAPITAL dans state.json au premier lancement.
            performance_data = calculate_performance_metrics(df_trades, initial_capital=initial_capital_for_calc)
        except Exception as e: 
            trades_data = [{"error": f"Erreur lecture trades: {str(e)}"}]
            performance_data = {"error": f"Erreur calcul métriques: {str(e)}"}
    else: 
        trades_data = []
        performance_data = calculate_performance_metrics(pd.DataFrame(), initial_capital=INITIAL_PAPER_CAPITAL)


    if os.path.exists(ACTIVITY_LOG_FILE):
        try:
            with open(ACTIVITY_LOG_FILE, 'r') as f: lines = f.readlines()
            activity_data = lines[-50:] # Les 50 dernières lignes
        except Exception as e: activity_data = [f"Erreur lecture log activité: {str(e)}"]
    else: activity_data = ["Fichier d'activité non trouvé."]

    return jsonify({
        "current_state": state_data,
        "trades_history": trades_data,
        "performance_summary": performance_data,
        "activity_log": activity_data
    })

@app.route('/api/kline_data_for_trade')
def get_kline_data_for_trade():
    entry_time_utc_str_arg = request.args.get('entry_time_utc') # Récupérer l'argument
    if not entry_time_utc_str_arg:
        return jsonify({"error": "Paramètre 'entry_time_utc' manquant."}), 400

    try:
        # 1. Trouver le trade correspondant dans TRADES_LOG_FILE
        if not os.path.exists(TRADES_LOG_FILE):
            return jsonify({"error": f"Fichier log des trades '{TRADES_LOG_FILE}' non trouvé."}), 404
        
        df_all_trades = pd.read_csv(TRADES_LOG_FILE, delimiter=';')
        # Convertir les colonnes de date au format datetime UTC pour comparaison PRÉCISE
        df_all_trades['EntryTimeUTC_dt'] = pd.to_datetime(df_all_trades['EntryTimeUTC'], errors='coerce', utc=True)
        
        # Le timestamp passé par JS sera probablement une string ISO. Convertir en datetime.
        target_entry_time = pd.to_datetime(entry_time_utc_str_arg, errors='coerce', utc=True)

        if pd.isna(target_entry_time):
            return jsonify({"error": "Format de 'entry_time_utc' invalide."}), 400

        # Trouver le trade. Comparer les objets datetime.
        trade_info_series = df_all_trades[df_all_trades['EntryTimeUTC_dt'] == target_entry_time]
        
        if trade_info_series.empty:
            logging.warning(f"Trade avec entrée à {target_entry_time} non trouvé. Essayez avec le format exact du CSV.")
            # Essayer de trouver en comparant les chaînes de caractères (moins robuste mais peut aider au débogage)
            trade_info_series = df_all_trades[df_all_trades['EntryTimeUTC'] == entry_time_utc_str_arg]
            if trade_info_series.empty:
                 return jsonify({"error": f"Trade avec entrée à '{entry_time_utc_str_arg}' non trouvé après plusieurs tentatives."}), 404
        
        trade_info = trade_info_series.iloc[0].to_dict()
        
        # 2. Définir la période pour charger les klines
        entry_dt = pd.to_datetime(trade_info['EntryTimeUTC'], utc=True) # Assurer que c'est un datetime
        exit_dt_from_log = trade_info.get('ExitTimeUTC')
        exit_dt = pd.to_datetime(exit_dt_from_log, utc=True) if pd.notnull(exit_dt_from_log) else entry_dt + timedelta(hours=48) # Fenêtre plus large si trade toujours ouvert

        max_indicator_period = max(BEST_PARAMS['sma_long'], BEST_PARAMS['adx_len'], BEST_PARAMS['rsi_len'], 200) 
        context_hours = max_indicator_period + 48 # Assez de contexte avant/après
        
        plot_start_dt = entry_dt - timedelta(hours=context_hours) 
        plot_end_dt = exit_dt + timedelta(hours=24) 

        # 3. Charger les klines brutes
        if not os.path.exists(LOCAL_KLINES_FILE):
            return jsonify({"error": f"Fichier klines local {LOCAL_KLINES_FILE} non trouvé."}), 404
        
        df_klines_full = pd.read_csv(LOCAL_KLINES_FILE, index_col='Open time', parse_dates=True, sep=';')
        if df_klines_full.index.tz is None: df_klines_full.index = df_klines_full.index.tz_localize('UTC')
        else: df_klines_full.index = df_klines_full.index.tz_convert('UTC')

        df_trade_period_klines = df_klines_full[
            (df_klines_full.index >= plot_start_dt) & 
            (df_klines_full.index <= plot_end_dt)
        ].copy()

        if df_trade_period_klines.empty or len(df_trade_period_klines) < max_indicator_period // 2 : # Seuil moins strict ici
            logging.warning(f"Pas assez de données klines ({len(df_trade_period_klines)}) pour période du trade {entry_dt} avec indicateurs.")
            # On peut quand même essayer de retourner les klines brutes si elles existent, même sans indicateurs complets
            if df_trade_period_klines.empty: return jsonify({"error": "Aucune donnée kline trouvée pour la période du trade."}), 404

        # 4. Recalculer les indicateurs
        df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_short"]}'] = df_trade_period_klines['Close'].rolling(window=BEST_PARAMS["sma_short"], min_periods=BEST_PARAMS["sma_short"]).mean()
        df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_long"]}'] = df_trade_period_klines['Close'].rolling(window=BEST_PARAMS["sma_long"], min_periods=BEST_PARAMS["sma_long"]).mean()
        df_trade_period_klines['SMA_200'] = df_trade_period_klines['Close'].rolling(window=200, min_periods=200).mean()
        adx_col = f"ADX_{BEST_PARAMS['adx_len']}"; rsi_col = f"RSI_{BEST_PARAMS['rsi_len']}"
        if hasattr(df_trade_period_klines, 'ta'):
            df_trade_period_klines.ta.adx(length=BEST_PARAMS['adx_len'], append=True, col_names=(adx_col, f"DMP_{BEST_PARAMS['adx_len']}",f"DMN_{BEST_PARAMS['adx_len']}"))
            df_trade_period_klines.ta.rsi(length=BEST_PARAMS['rsi_len'], append=True, col_names=(rsi_col,))
        
        # 5. Récupérer les valeurs des indicateurs au moment de l'entrée et de la sortie
        conditions_at_entry = {}; conditions_at_exit = {}
        entry_candle_data = df_trade_period_klines.loc[df_trade_period_klines.index.asof(entry_dt)] if entry_dt in df_trade_period_klines.index else None
        if entry_candle_data is not None:
            conditions_at_entry = {k: (entry_candle_data.get(k) if pd.notnull(entry_candle_data.get(k)) else None) for k in [adx_col, rsi_col, f'SMA_{BEST_PARAMS["sma_short"]}', f'SMA_{BEST_PARAMS["sma_long"]}', 'Close']}
        
        if pd.notnull(exit_dt) and exit_dt in df_trade_period_klines.index:
            exit_candle_data = df_trade_period_klines.loc[df_trade_period_klines.index.asof(exit_dt)]
            conditions_at_exit = {k: (exit_candle_data.get(k) if pd.notnull(exit_candle_data.get(k)) else None) for k in [adx_col, rsi_col, f'SMA_{BEST_PARAMS["sma_short"]}', f'SMA_{BEST_PARAMS["sma_long"]}', 'Close']}

        chart_data = {
            "dates": df_trade_period_klines.index.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(), # Format ISO pour JS
            "open": [x if pd.notnull(x) else None for x in df_trade_period_klines['Open'].tolist()],
            "high": [x if pd.notnull(x) else None for x in df_trade_period_klines['High'].tolist()],
            "low": [x if pd.notnull(x) else None for x in df_trade_period_klines['Low'].tolist()],
            "close": [x if pd.notnull(x) else None for x in df_trade_period_klines['Close'].tolist()],
            "volume": [x if pd.notnull(x) else None for x in df_trade_period_klines['Volume'].tolist()],
            "sma_short": [x if pd.notnull(x) else None for x in df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_short"]}'].tolist()],
            "sma_long": [x if pd.notnull(x) else None for x in df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_long"]}'].tolist()],
            "sma_200": [x if pd.notnull(x) else None for x in df_trade_period_klines['SMA_200'].tolist()],
            "adx": [x if pd.notnull(x) else None for x in df_trade_period_klines[adx_col].tolist()],
            "rsi": [x if pd.notnull(x) else None for x in df_trade_period_klines[rsi_col].tolist()],
            "trade_info": {
                "entry_price": trade_info.get('EntryPrice'),
                "entry_time_utc": entry_dt.isoformat(),
                "exit_price": trade_info.get('ExitPrice'),
                "exit_time_utc": exit_dt.isoformat() if pd.notnull(exit_dt) else None,
                "sl_price": float(trade_info.get('EntryPrice')) * (1 - BEST_PARAMS['sl']) if pd.notnull(trade_info.get('EntryPrice')) else None,
                "tp_price": float(trade_info.get('EntryPrice')) * (1 + BEST_PARAMS['tp']) if pd.notnull(trade_info.get('EntryPrice')) else None,
                "exit_reason": trade_info.get('ExitReason')
            },
            "conditions_at_entry": conditions_at_entry,
            "conditions_at_exit": conditions_at_exit
        }
        return jsonify(chart_data)

    except FileNotFoundError as e:
        logging.error(f"Fichier non trouvé dans /api/kline_data_for_trade: {e}")
        return jsonify({"error": f"Fichier de données non trouvé sur le serveur: {e.filename}"}), 500
    except KeyError as e:
        logging.error(f"Clé manquante lors de l'accès aux données du trade ou klines: {e}")
        return jsonify({"error": f"Données de trade ou klines incomplètes. Clé manquante: {e}"}), 500
    except Exception as e:
        logging.exception("Erreur dans /api/kline_data_for_trade") # Log traceback complet côté serveur
        return jsonify({"error": f"Erreur serveur inattendue: {str(e)}"}), 500

# Fonction utilitaire pour calculer les indicateurs sur un DataFrame donné
def add_strategy_indicators(df_market_data, params):
    df_with_indicators = df_market_data.copy()
    
    # S'assurer que les colonnes nécessaires existent
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_with_indicators.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_with_indicators.columns]
        logging.error(f"Colonnes manquantes pour add_strategy_indicators: {missing} dans le DataFrame avec index: {df_with_indicators.index.name} et colonnes: {df_with_indicators.columns}")
        return pd.DataFrame() # Retourner vide si colonnes essentielles manquent

    # Calculer SMA_200 si elle n'est pas déjà là 
    if 'SMA_200' not in df_with_indicators.columns or df_with_indicators['SMA_200'].isnull().all():
        df_with_indicators['SMA_200'] = df_with_indicators['Close'].rolling(window=200, min_periods=200).mean()
    
    adx_col = f"ADX_{params['adx_len']}"; rsi_col = f"RSI_{params['rsi_len']}"
    sma_s_col = f"SMA_{params['sma_short']}"; sma_l_col = f"SMA_{params['sma_long']}"

    if hasattr(df_with_indicators, 'ta'):
        # Calculer seulement si la colonne n'existe pas ou est pleine de NaN
        if adx_col not in df_with_indicators.columns or df_with_indicators[adx_col].isnull().all():
            df_with_indicators.ta.adx(length=params['adx_len'], append=True, col_names=(adx_col, f"DMP_{params['adx_len']}",f"DMN_{params['adx_len']}"))
        if rsi_col not in df_with_indicators.columns or df_with_indicators[rsi_col].isnull().all():
            df_with_indicators.ta.rsi(length=params['rsi_len'], append=True, col_names=(rsi_col,))
        if sma_s_col not in df_with_indicators.columns or df_with_indicators[sma_s_col].isnull().all():
            df_with_indicators[sma_s_col] = df_with_indicators['Close'].rolling(window=params['sma_short'], min_periods=params['sma_short']).mean()
        if sma_l_col not in df_with_indicators.columns or df_with_indicators[sma_l_col].isnull().all():
            df_with_indicators[sma_l_col] = df_with_indicators['Close'].rolling(window=params['sma_long'], min_periods=params['sma_long']).mean()
    else:
        logging.error("Pandas TA non disponible pour add_strategy_indicators."); return pd.DataFrame()
    
    # Pas besoin de dropna ici, on veut retourner toutes les données avec les indicateurs (même avec des NaN au début)
    return df_with_indicators


@app.route('/api/current_market_data_with_indicators')
def get_current_market_data():
    try:
        # Récupérer les HISTORICAL_DATA_FETCH_LIMIT dernières bougies
        df_klines = get_market_klines(limit=HISTORICAL_DATA_FETCH_LIMIT) 
        if df_klines.empty or len(df_klines) < max(BEST_PARAMS['sma_long'], BEST_PARAMS['adx_len'], 200) : # Besoin d'assez pour les indicateurs de base
            return jsonify({"error": "Pas assez de données klines récentes pour calculer les indicateurs."}), 500

        df_with_indicators = add_strategy_indicators(df_klines, BEST_PARAMS)
        
        if df_with_indicators.empty:
            return jsonify({"error": "Erreur lors du calcul des indicateurs sur les données récentes."}), 500
            
        # Récupérer les dernières N bougies (ex: 100) pour l'affichage après calcul des indicateurs
        display_limit = 100 
        df_display = df_with_indicators.iloc[-display_limit:] if len(df_with_indicators) > display_limit else df_with_indicators
        
        # Calculer le signal de régime pour la dernière bougie disponible
        latest_signal = 0
        if not df_display.empty:
            # Pour calculer le signal de régime, on a besoin des signaux SMA et RSI
            temp_sma_sig = generate_sma_crossover_signals(df_display.copy(), sma_short_len=BEST_PARAMS['sma_short'], sma_long_len=BEST_PARAMS['sma_long'])
            temp_rsi_sig = generate_simple_rsi_signals(df_display.copy(), rsi_length=BEST_PARAMS['rsi_len'], 
                                                       rsi_oversold=BEST_PARAMS['rsi_os'], 
                                                       rsi_overbought_exit=BEST_PARAMS['rsi_ob_exit'])
            
            # Rejoindre les signaux au df_display principal.
            df_display = df_display.join(temp_sma_sig[['signal_SmaCross']], how='left')
            df_display = df_display.join(temp_rsi_sig[['signal_SimpleRsi']], how='left')
            df_display.fillna({'signal_SmaCross': 0, 'signal_SimpleRsi': 0}, inplace=True)

            adx_col = f"ADX_{BEST_PARAMS['adx_len']}"
            if adx_col in df_display.columns and not df_display.empty:
                latest_adx_val = df_display[adx_col].iloc[-1]
                if latest_adx_val > BEST_PARAMS['adx_trend_th']:
                    latest_signal = df_display['signal_SmaCross'].iloc[-1]
                elif latest_adx_val < BEST_PARAMS['adx_range_th']:
                    latest_signal = df_display['signal_SimpleRsi'].iloc[-1]
        
        chart_data = {
            "dates": df_display.index.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
            "open": [x if pd.notnull(x) else None for x in df_display['Open'].tolist()],
            "high": [x if pd.notnull(x) else None for x in df_display['High'].tolist()],
            "low": [x if pd.notnull(x) else None for x in df_display['Low'].tolist()],
            "close": [x if pd.notnull(x) else None for x in df_display['Close'].tolist()],
            "volume": [x if pd.notnull(x) else None for x in df_display['Volume'].tolist()],
            "sma_short": [x if pd.notnull(x) else None for x in df_display[f'SMA_{BEST_PARAMS["sma_short"]}'].tolist()],
            "sma_long": [x if pd.notnull(x) else None for x in df_display[f'SMA_{BEST_PARAMS["sma_long"]}'].tolist()],
            "sma_200": [x if pd.notnull(x) else None for x in df_display['SMA_200'].tolist()],
            "adx": [x if pd.notnull(x) else None for x in df_display[f"ADX_{BEST_PARAMS['adx_len']}"].tolist()],
            "rsi": [x if pd.notnull(x) else None for x in df_display[f"RSI_{BEST_PARAMS['rsi_len']}"].tolist()],
            "latest_indicators": { # Valeurs de la dernière bougie
                "price": df_display['Close'].iloc[-1] if not df_display.empty else None,
                "adx": df_display[f"ADX_{BEST_PARAMS['adx_len']}"].iloc[-1] if not df_display.empty and f"ADX_{BEST_PARAMS['adx_len']}" in df_display else None,
                "rsi": df_display[f"RSI_{BEST_PARAMS['rsi_len']}"].iloc[-1] if not df_display.empty and f"RSI_{BEST_PARAMS['rsi_len']}" in df_display else None,
                "sma_short": df_display[f"SMA_{BEST_PARAMS['sma_short']}"].iloc[-1] if not df_display.empty and f"SMA_{BEST_PARAMS['sma_short']}" in df_display else None,
                "sma_long": df_display[f"SMA_{BEST_PARAMS['sma_long']}"].iloc[-1] if not df_display.empty and f"SMA_{BEST_PARAMS['sma_long']}" in df_display else None,
                "signal_sma_cross": int(df_display['signal_SmaCross'].iloc[-1]) if not df_display.empty and 'signal_SmaCross' in df_display else 0,
                "signal_rsi": int(df_display['signal_SimpleRsi'].iloc[-1]) if not df_display.empty and 'signal_SimpleRsi' in df_display else 0,
                "final_signal_regime": int(latest_signal)
            }
        }
        return jsonify(chart_data)

    except Exception as e:
        logging.exception("Erreur dans /api/current_market_data_with_indicators")
        return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500


if __name__ == '__main__':
    # Créer des fichiers vides avec en-tête si inexistants
    if not os.path.exists(STATE_FILE):
        initial_state = {"in_position": False, "entry_price": 0.0, "quantity": 0.0, 
                         "sl_price": 0.0, "tp_price": 0.0, "entry_time_utc_str": None, 
                         "paper_capital": INITIAL_PAPER_CAPITAL, 
                         "last_known_equity": INITIAL_PAPER_CAPITAL,
                         "paper_capital_at_start_of_log": INITIAL_PAPER_CAPITAL} # Important pour calculate_performance_metrics
        
        # Sauvegarder l'état initial (s'assurer que la fonction save_state est définie ou importée)
        # Pour ce script, on peut la définir ici rapidement si elle n'est pas importée
        def temp_save_state(state, filename=STATE_FILE):
            try:
                state_to_save = state.copy()
                if state_to_save.get("entry_time_utc") and isinstance(state_to_save["entry_time_utc"], datetime):
                    state_to_save["entry_time_utc_str"] = state_to_save["entry_time_utc"].isoformat()
                    if "entry_time_utc" in state_to_save: del state_to_save["entry_time_utc"]
                else: state_to_save["entry_time_utc_str"] = None
                with open(filename, 'w') as f: json.dump(state_to_save, f, indent=4)
            except Exception as e: print(f"Erreur temp_save_state: {e}")
        temp_save_state(initial_state)


    if not os.path.exists(TRADES_LOG_FILE):
        with open(TRADES_LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['EntryTimeUTC', 'EntryPrice', 'ExitTimeUTC', 'ExitPrice', 'Quantity', 'PnL_Abs', 'PnL_Pct', 'ExitReason', 'CommissionTotal', 'CapitalAvant', 'CapitalApres'])
    
    app.run(debug=True, host='0.0.0.0', port=5001)
