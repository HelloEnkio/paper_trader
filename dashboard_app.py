    # dashboard_app.py
from flask import Flask, jsonify, render_template
import pandas as pd
import pandas_ta as ta # Nécessaire pour recalculer les indicateurs
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta

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


# Paramètres de la stratégie (BEST_PARAMS de votre paper_trader.py)
BEST_PARAMS = {
    'adx_len': 20, 'adx_trend_th': 28, 'adx_range_th': 22,
    'sma_short': 30, 'sma_long': 50, 
    'rsi_len': 10, 'rsi_os': 35, 'rsi_ob_exit': 65,
    'sl': 0.02, 'tp': 0.088 
}
INITIAL_PAPER_CAPITAL = 50.0 # Doit correspondre à celui de votre paper_trader.py

app = Flask(__name__)

# --- Fonctions Utilitaires (copiées/adaptées de vos scripts précédents) ---
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
        return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": np.nan, "num_trades": 0, 
                "win_rate_pct": 0, "profit_factor": 0, "current_equity": initial_capital, "pnl_total_abs":0,
                "equity_curve_values": [initial_capital], "equity_curve_timestamps": [None]} # Timestamp initial à None

    current_equity = initial_capital
    equity_over_time_values = [initial_capital]
    # Utiliser les timestamps de sortie des trades pour l'axe X de la courbe d'équité
    # Le premier point est le capital initial avant tout trade.
    equity_over_time_timestamps = [df_trades['EntryTimeUTC'].iloc[0] if not df_trades.empty else None] # Timestamp du début du premier trade

    df_trades['PnL_Abs'] = pd.to_numeric(df_trades['PnL_Abs'], errors='coerce').fillna(0)
    df_trades['ExitTimeUTC'] = pd.to_datetime(df_trades['ExitTimeUTC'], errors='coerce')


    for idx, row in df_trades.iterrows():
        current_equity += row['PnL_Abs']
        equity_over_time_values.append(current_equity)
        equity_over_time_timestamps.append(row['ExitTimeUTC']) # Timestamp de sortie du trade
    
    equity_series = pd.Series(equity_over_time_values, index=pd.to_datetime(equity_over_time_timestamps, utc=True))
    # S'il y a des NaT dans l'index (par ex. le premier None), on les gère
    equity_series = equity_series[equity_series.index.notna()]
    if equity_series.empty: # Si après le filtre, c'est vide (devrait pas arriver si trades)
         return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": np.nan, "num_trades": 0, 
                "win_rate_pct": 0, "profit_factor": 0, "current_equity": initial_capital, "pnl_total_abs":0,
                "equity_curve_values": [initial_capital], "equity_curve_timestamps": [None]}


    total_return_pct = (current_equity / initial_capital - 1) if initial_capital > 0 else 0
    
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series / peak) - 1
    max_drawdown_pct = drawdown.min()
    
    # Calcul du Sharpe Ratio basé sur les rendements périodiques de l'équité (trade par trade)
    equity_periodic_returns = equity_series.pct_change().fillna(0.0)
    sharpe_ratio = np.nan
    if not equity_periodic_returns.empty and equity_periodic_returns.std(ddof=0) != 0:
        # L'annualisation est délicate ici sans connaître la fréquence exacte des trades.
        # On peut faire un Sharpe simple sur les rendements des trades, ou estimer.
        # Pour un Sharpe Ratio annualisé, il faudrait des rendements à fréquence fixe (ex: quotidiens).
        # On peut calculer un Sharpe "par trade" pour l'instant.
        # Ou si les trades sont assez fréquents, on peut annualiser grossièrement.
        # Supposons une durée moyenne de trade pour annualiser (besoin de timestamps pour ça)
        # Pour l'instant, calculons un Sharpe simplifié non annualisé ou mettons NaN.
        try:
            # Si on a assez de trades pour une estimation
            if len(equity_periodic_returns) > 20 : # Au moins 20 trades pour une stat un peu stable
                 # Supposons que les trades sont la "période"
                sharpe_ratio = (equity_periodic_returns.mean() / equity_periodic_returns.std(ddof=0)) * np.sqrt(len(equity_periodic_returns)) # sqrt(N) pour annualiser grossièrement
        except ZeroDivisionError:
            sharpe_ratio = np.nan


    num_trades = len(df_trades)
    winning_trades = df_trades[df_trades['PnL_Abs'] > 0]
    win_rate_pct = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0
    
    total_profit = df_trades[df_trades['PnL_Abs'] > 0]['PnL_Abs'].sum()
    total_loss = abs(df_trades[df_trades['PnL_Abs'] < 0]['PnL_Abs'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    return {
        "total_return_pct": total_return_pct * 100,
        "max_drawdown_pct": max_drawdown_pct * 100,
        "sharpe_ratio": sharpe_ratio, 
        "num_trades": num_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "current_equity": current_equity,
        "pnl_total_abs": current_equity - initial_capital,
        "equity_curve_values": equity_series.tolist(),
        "equity_curve_timestamps": [ts.isoformat() if pd.notnull(ts) else None for ts in equity_series.index.tolist()]
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
    entry_time_str = request.args.get('entry_time_utc') # ex: 2025-05-30T04:00:00+00:00
    if not entry_time_str:
        return jsonify({"error": "Timestamp d'entrée du trade (entry_time_utc) manquant."}), 400

    try:
        # 1. Trouver le trade correspondant dans TRADES_LOG_FILE
        df_all_trades = pd.read_csv(TRADES_LOG_FILE, delimiter=';', parse_dates=['EntryTimeUTC', 'ExitTimeUTC'])
        # S'assurer que les dates sont UTC pour la comparaison
        df_all_trades['EntryTimeUTC'] = pd.to_datetime(df_all_trades['EntryTimeUTC'], utc=True)
        target_entry_time = pd.to_datetime(entry_time_str, utc=True)
        
        trade_info = df_all_trades[df_all_trades['EntryTimeUTC'] == target_entry_time]
        if trade_info.empty:
            return jsonify({"error": f"Trade avec entrée à {entry_time_str} non trouvé."}), 404
        
        trade_info = trade_info.iloc[0].to_dict()
        
        # 2. Définir la période pour charger les klines (autour du trade)
        # S'assurer que entry_time et exit_time sont bien des objets datetime
        entry_dt = pd.to_datetime(trade_info['EntryTimeUTC'], utc=True)
        exit_dt = pd.to_datetime(trade_info['ExitTimeUTC'], utc=True) if pd.notnull(trade_info.get('ExitTimeUTC')) else entry_dt + timedelta(hours=24) # Default window if no exit

        # Période pour afficher les klines: un peu avant l'entrée, un peu après la sortie
        # Le nombre de bougies avant/après dépend de la plus longue période d'indicateur (ex: SMA50 ou ADX20)
        lookback_candles = max(BEST_PARAMS['sma_long'], BEST_PARAMS['adx_len'], BEST_PARAMS['rsi_len'], 50) + 10
        plot_start_dt = entry_dt - timedelta(hours=lookback_candles) 
        plot_end_dt = exit_dt + timedelta(hours=20) 

        # 3. Charger les klines brutes depuis le fichier local
        if not os.path.exists(LOCAL_KLINES_FILE):
            return jsonify({"error": f"Fichier klines local {LOCAL_KLINES_FILE} non trouvé."}), 404
        
        df_klines_full = pd.read_csv(LOCAL_KLINES_FILE, index_col='Open time', parse_dates=True, sep=';')
        if df_klines_full.index.tz is None: df_klines_full.index = df_klines_full.index.tz_localize('UTC')
        else: df_klines_full.index = df_klines_full.index.tz_convert('UTC')

        df_trade_period_klines = df_klines_full[
            (df_klines_full.index >= plot_start_dt) & 
            (df_klines_full.index <= plot_end_dt)
        ].copy()

        if df_trade_period_klines.empty:
            return jsonify({"error": "Aucune donnée kline trouvée pour la période du trade."}), 404

        # 4. Recalculer les indicateurs pour cette période
        df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_short"]}'] = df_trade_period_klines['Close'].rolling(window=BEST_PARAMS["sma_short"], min_periods=BEST_PARAMS["sma_short"]).mean()
        df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_long"]}'] = df_trade_period_klines['Close'].rolling(window=BEST_PARAMS["sma_long"], min_periods=BEST_PARAMS["sma_long"]).mean()
        adx_col = f"ADX_{BEST_PARAMS['adx_len']}"; rsi_col = f"RSI_{BEST_PARAMS['rsi_len']}"
        if hasattr(df_trade_period_klines, 'ta'):
            df_trade_period_klines.ta.adx(length=BEST_PARAMS['adx_len'], append=True, col_names=(adx_col, f"DMP_{BEST_PARAMS['adx_len']}",f"DMN_{BEST_PARAMS['adx_len']}"))
            df_trade_period_klines.ta.rsi(length=BEST_PARAMS['rsi_len'], append=True, col_names=(rsi_col,))
        
        # Préparer les données pour Plotly.js (ou autre biblio de graphiques JS)
        # Plotly attend des listes pour chaque trace (Open, High, Low, Close, dates)
        chart_data = {
            "dates": df_trade_period_klines.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "open": df_trade_period_klines['Open'].tolist(),
            "high": df_trade_period_klines['High'].tolist(),
            "low": df_trade_period_klines['Low'].tolist(),
            "close": df_trade_period_klines['Close'].tolist(),
            "volume": df_trade_period_klines['Volume'].tolist(),
            "sma_short": df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_short"]}'].fillna(None).tolist(), # Remplacer NaN par None pour JSON
            "sma_long": df_trade_period_klines[f'SMA_{BEST_PARAMS["sma_long"]}'].fillna(None).tolist(),
            "adx": df_trade_period_klines[adx_col].fillna(None).tolist(),
            "rsi": df_trade_period_klines[rsi_col].fillna(None).tolist(),
            "trade_info": { # Ajouter les infos du trade pour les marqueurs
                "entry_price": trade_info.get('EntryPrice'),
                "entry_time": pd.to_datetime(trade_info.get('EntryTimeUTC')).strftime('%Y-%m-%d %H:%M:%S'),
                "exit_price": trade_info.get('ExitPrice'),
                "exit_time": pd.to_datetime(trade_info.get('ExitTimeUTC')).strftime('%Y-%m-%d %H:%M:%S'),
                "sl_price": BEST_PARAMS['sl'] * trade_info.get('EntryPrice') if trade_info.get('EntryPrice') else None, # Recalculer pour affichage
                "tp_price": BEST_PARAMS['tp'] * trade_info.get('EntryPrice') if trade_info.get('EntryPrice') else None, # Recalculer pour affichage
            }
        }
        return jsonify(chart_data)

    except Exception as e:
        logging.exception("Erreur dans /api/kline_data_for_trade")
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
