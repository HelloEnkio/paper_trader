    # dashboard_app.py
from flask import Flask, jsonify, render_template, request
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
INITIAL_PAPER_CAPITAL = 44.42 # Doit correspondre à celui de votre paper_trader.py

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
