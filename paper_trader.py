import os
import time
import csv
import requests  # Assurez-vous que cette librairie est installée
import hashlib
import hmac
import base64
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import json
from datetime import datetime, timezone, timedelta
import schedule  # Assurez-vous que cette librairie est installée

# --- Charger les Variables d'Environnement ---
# Pour une meilleure pratique, utilisez python-dotenv
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('KUCOIN_API_KEY')
API_SECRET = os.getenv('KUCOIN_API_SECRET')
API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE')


BASE_URL = 'https://api.kucoin.com'
SYMBOL = "ETH-USDT"
TIMEFRAME_KUCOIN = '1hour'
HISTORICAL_DATA_FETCH_LIMIT = 350

# --- Paramètres de la Stratégie Finale (BEST_PARAMS) ---
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

# --- Configuration du Paper Trading ---
INITIAL_PAPER_CAPITAL = 44.42
RISK_PER_TRADE_PCT = 0.02
COMMISSION_RATE = 0.002  # Taker fee 0.2% pour être conservateur

STATE_FILE = "paper_bot_state.json"
TRADES_LOG_FILE = "paper_bot_trades.csv"
KLINES_LOG_FILE = f"local_klines_{SYMBOL.replace('-', '_')}_{TIMEFRAME_KUCOIN}.csv"
LOG_FILE = "paper_bot_activity.log"

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()])


# --- Fonctions d'Interaction API KuCoin ---
def kucoin_request_signature(method, path, params_for_sign_string=None, body_params=None):
    """Génère la signature pour les requêtes KuCoin authentifiées."""
    now_ms = str(int(time.time() * 1000))
    str_to_sign = now_ms + method.upper() + path

    if params_for_sign_string:  # Pour GET (query params)
        str_to_sign += "?" + requests.compat.urlencode(params_for_sign_string)
    elif body_params:  # Pour POST (json body)
        str_to_sign += json.dumps(body_params)

    signature = base64.b64encode(
        hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')

    headers = {
        "KC-API-KEY": API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": now_ms,
        "KC-API-PASSPHRASE": API_PASSPHRASE,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }
    return headers


def get_market_klines(symbol=SYMBOL, timeframe_type=TIMEFRAME_KUCOIN, limit=HISTORICAL_DATA_FETCH_LIMIT):
    path = '/api/v1/market/candles'
    # KuCoin prend le timestamp en secondes pour startAt/endAt
    end_time_s = int(time.time())
    start_time_s = end_time_s - (limit * 3600)  # 3600 secondes par heure pour '1hour'

    params = {'symbol': symbol, 'type': timeframe_type, 'startAt': start_time_s}
    logging.info(
        f"Récupération de ~{limit} bougies pour {symbol} type {timeframe_type} depuis {datetime.fromtimestamp(start_time_s, tz=timezone.utc)}")

    try:
        # Les klines publiques n'ont pas besoin de signature, mais certains brokers l'exigent quand même ou c'est plus simple de l'avoir
        # Pour KuCoin, les klines publiques fonctionnent sans signature.
        response = requests.get(BASE_URL + path, params=params)
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

            # KuCoin peut retourner plus de données que 'limit' si startAt est ancien. On prend les dernières.
            if len(df) > limit:
                df = df.iloc[-limit:]

            save_klines_locally(df, symbol, timeframe_type)
            return df
        else:
            logging.error(
                f"Erreur API KuCoin (klines): {response.status_code} - {response.text} pour GET {path} avec params {params}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de requête (klines) vers KuCoin: {e}")
        return pd.DataFrame()


def save_klines_locally(df_new_klines, symbol, timeframe):
    filename = f"local_klines_{symbol.replace('-', '_')}_{timeframe}.csv"
    df_to_save = df_new_klines.copy()
    # S'assurer que l'index a un nom pour que to_csv fonctionne bien si l'index est sauvegardé
    if df_to_save.index.name is None:
        df_to_save.index.name = 'Open time'

    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename, index_col='Open time', parse_dates=True, sep=';')
            if df_existing.index.tz is None:
                df_existing.index = df_existing.index.tz_localize('UTC')
            else:
                df_existing.index = df_existing.index.tz_convert('UTC')

            if df_to_save.index.tz is None:
                df_to_save.index = df_to_save.index.tz_localize('UTC')
            else:
                df_to_save.index = df_to_save.index.tz_convert('UTC')

            df_combined = pd.concat([df_existing, df_to_save])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined.sort_index(inplace=True)
            df_combined.to_csv(filename, sep=';')  # Sauvegarde avec index
            logging.info(f"Données de marché mises à jour dans {filename} ({len(df_combined)} lignes).")
        except Exception as e:
            logging.error(f"Erreur maj fichier klines local {filename}: {e}")
            df_to_save.to_csv(filename, sep=';')  # Écrase avec les nouvelles données en cas d'erreur
            logging.info(f"Nouvelles données de marché sauvegardées (écrasement) dans {filename}.")
    else:
        df_to_save.to_csv(filename, sep=';')  # Sauvegarde avec index
        logging.info(f"Données de marché initiales sauvegardées dans {filename}.")


# --- Fonctions de Génération de Signaux ---
def generate_sma_crossover_signals(df_period_in, sma_short_len, sma_long_len):
    df_p = df_period_in.copy()
    sma_short_col = f'SMA_{sma_short_len}';
    sma_long_col = f'SMA_{sma_long_len}'
    if sma_short_col not in df_p.columns or df_p[sma_short_col].isnull().all():
        df_p[sma_short_col] = df_p['Close'].rolling(window=sma_short_len, min_periods=sma_short_len).mean()
    if sma_long_col not in df_p.columns or df_p[sma_long_col].isnull().all():
        df_p[sma_long_col] = df_p['Close'].rolling(window=sma_long_len, min_periods=sma_long_len).mean()
    df_p.dropna(subset=[sma_short_col, sma_long_col], inplace=True)
    if df_p.empty: return df_p.assign(signal_SmaCross=0)
    df_p['position_SmaCross'] = np.where(df_p[sma_short_col] > df_p[sma_long_col], 1, -1)
    df_p['signal_SmaCross'] = 0
    diff_values = df_p['position_SmaCross'].diff().fillna(0)
    df_p.loc[diff_values == 2, 'signal_SmaCross'] = 1
    df_p.loc[diff_values == -2, 'signal_SmaCross'] = -1
    return df_p


def generate_simple_rsi_signals(df_period_in, rsi_length, rsi_oversold, rsi_overbought_exit):
    df_p = df_period_in.copy()
    rsi_col_name = f'RSI_{rsi_length}'
    if rsi_col_name not in df_p.columns or df_p[rsi_col_name].isnull().all():
        if hasattr(df_p, 'ta'):
            df_p.ta.rsi(length=rsi_length, append=True, col_names=(rsi_col_name,))
            df_p.dropna(subset=[rsi_col_name], inplace=True)
        else:
            return df_p.assign(signal_SimpleRsi=0)
    if df_p.empty or rsi_col_name not in df_p.columns: return df_p.assign(signal_SimpleRsi=0)
    df_p['signal_SimpleRsi'] = 0
    df_p.loc[(df_p[rsi_col_name].shift(1).fillna(50) < rsi_oversold) & (
                df_p[rsi_col_name] >= rsi_oversold), 'signal_SimpleRsi'] = 1
    df_p.loc[(df_p[rsi_col_name].shift(1).fillna(50) > rsi_overbought_exit) & (
                df_p[rsi_col_name] <= rsi_overbought_exit), 'signal_SimpleRsi'] = -1
    return df_p


# --- Gestion de l'État ---
def load_state(filename=STATE_FILE):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            if state.get("entry_time_utc_str"):  # J'avais sauvegardé en str
                state["entry_time_utc"] = datetime.fromisoformat(state["entry_time_utc_str"])
            else:
                state["entry_time_utc"] = None
            state.setdefault("paper_capital", INITIAL_PAPER_CAPITAL)
            state.setdefault("last_known_equity", state.get("paper_capital", INITIAL_PAPER_CAPITAL))
            state.setdefault("in_position", False)
            state.setdefault("entry_price", 0.0)
            state.setdefault("quantity", 0.0)
            state.setdefault("sl_price", 0.0)
            state.setdefault("tp_price", 0.0)
            return state
        except Exception as e:
            logging.error(f"Erreur chargement état {filename}: {e}")
    return {"in_position": False, "entry_price": 0.0, "quantity": 0.0, "sl_price": 0.0, "tp_price": 0.0,
            "entry_time_utc": None, "paper_capital": INITIAL_PAPER_CAPITAL, "last_known_equity": INITIAL_PAPER_CAPITAL}


def save_state(state, filename=STATE_FILE):
    try:
        state_to_save = state.copy()
        if state_to_save.get("entry_time_utc") and isinstance(state_to_save["entry_time_utc"], datetime):
            state_to_save["entry_time_utc_str"] = state_to_save["entry_time_utc"].isoformat()  # Sauvegarder en str
            del state_to_save["entry_time_utc"]  # Retirer l'objet datetime non sérialisable directement
        else:
            state_to_save["entry_time_utc_str"] = None

        with open(filename, 'w') as f:
            json.dump(state_to_save, f, indent=4)
    except Exception as e:
        logging.error(f"Erreur sauvegarde état {filename}: {e}")


def log_trade_to_csv(trade_details, filename=TRADES_LOG_FILE):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists: writer.writerow(
            ['EntryTimeUTC', 'EntryPrice', 'ExitTimeUTC', 'ExitPrice', 'Quantity', 'PnL_Abs', 'PnL_Pct', 'ExitReason',
             'CommissionTotal', 'CapitalAvant', 'CapitalApres'])
        writer.writerow([
            trade_details.get('entry_time_utc'), trade_details.get('entry_price'),
            trade_details.get('exit_time_utc'), trade_details.get('exit_price'),
            f"{trade_details.get('quantity', 0):.8f}",
            f"{trade_details.get('pnl_abs', 0):.2f}", f"{trade_details.get('pnl_pct', 0):.4%}",
            trade_details.get('exit_reason'), f"{trade_details.get('commission_total', 0):.4f}",
            f"{trade_details.get('capital_before_trade', 0):.2f}",
            f"{trade_details.get('capital_after_trade', 0):.2f}"])


# --- Boucle Principale du Bot de Paper Trading ---
def trading_bot_logic():
    logging.info("--- Début Itération Bot Paper Trading ---")
    current_state = load_state()

    logging.info(f"Récupération des {HISTORICAL_DATA_FETCH_LIMIT} dernières bougies pour {SYMBOL}...")
    df_market_raw = get_market_klines(symbol=SYMBOL, timeframe_type=TIMEFRAME_KUCOIN, limit=HISTORICAL_DATA_FETCH_LIMIT)

    min_rows_needed = max(200, BEST_PARAMS['adx_len'], BEST_PARAMS['sma_long'], BEST_PARAMS['rsi_len']) + 5  # Marge
    if df_market_raw.empty or len(df_market_raw) < min_rows_needed:
        logging.warning(
            f"Données de marché insuffisantes ({len(df_market_raw)} lignes, besoin d'au moins {min_rows_needed}). Prochaine tentative.")
        return

    df_market = df_market_raw.copy()
    df_market['SMA_200'] = df_market['Close'].rolling(window=200, min_periods=200).mean()
    adx_col = f"ADX_{BEST_PARAMS['adx_len']}";
    rsi_col = f"RSI_{BEST_PARAMS['rsi_len']}"
    if hasattr(df_market, 'ta'):
        df_market.ta.adx(length=BEST_PARAMS['adx_len'], append=True)
        df_market.ta.rsi(length=BEST_PARAMS['rsi_len'], append=True, col_names=(rsi_col,))
    else:
        logging.error("Pandas TA non disponible."); return

    # Dropna après TOUS les indicateurs pour cette itération
    df_market.dropna(inplace=True)
    if df_market.empty or len(df_market) < 2:
        logging.warning("DataFrame vide après calcul indicateurs. Prochaine tentative.")
        return

    df_market_sma_signals = generate_sma_crossover_signals(df_market.copy(), sma_short_len=BEST_PARAMS['sma_short'],
                                                           sma_long_len=BEST_PARAMS['sma_long'])
    df_market_rsi_signals = generate_simple_rsi_signals(df_market.copy(), rsi_length=BEST_PARAMS['rsi_len'],
                                                        rsi_oversold=BEST_PARAMS['rsi_os'],
                                                        rsi_overbought_exit=BEST_PARAMS['rsi_ob_exit'])

    df_market = df_market.join(df_market_sma_signals[['signal_SmaCross']], how='left')
    df_market = df_market.join(df_market_rsi_signals[['signal_SimpleRsi']], how='left')
    df_market.fillna({'signal_SmaCross': 0, 'signal_SimpleRsi': 0}, inplace=True)

    df_market['signal_RegimeSwitch'] = 0
    condition_trending = df_market[adx_col] > BEST_PARAMS['adx_trend_th']
    condition_ranging = df_market[adx_col] < BEST_PARAMS['adx_range_th']
    df_market.loc[condition_trending, 'signal_RegimeSwitch'] = df_market.loc[condition_trending, 'signal_SmaCross']
    df_market.loc[condition_ranging, 'signal_RegimeSwitch'] = df_market.loc[condition_ranging, 'signal_SimpleRsi']

    latest_data = df_market.iloc[-1]
    current_price = latest_data['Close'];
    current_low = latest_data['Low'];
    current_high = latest_data['High']
    final_signal = latest_data['signal_RegimeSwitch'];
    current_time_utc = df_market.index[-1]

    logging.info(
        f"[{current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}] Prix: {current_price:.2f}, ADX: {latest_data[adx_col]:.2f}, Signal: {final_signal}, Position: {'LONG (' + str(current_state['quantity']) + ')' if current_state['in_position'] else 'NEUTRE'}")

    capital_before_action = current_state['paper_capital']

    if current_state['in_position']:
        exit_price_simulated = None;
        exit_reason_simulated = None
        if current_low <= current_state['sl_price']:
            exit_price_simulated = current_state['sl_price']; exit_reason_simulated = "SL"
        elif current_high >= current_state['tp_price']:
            exit_price_simulated = current_state['tp_price']; exit_reason_simulated = "TP"
        elif final_signal == -1:
            exit_price_simulated = current_price; exit_reason_simulated = "Signal"

        if exit_price_simulated is not None:
            logging.info(
                f"ORDRE DE VENTE SIMULÉ ({exit_reason_simulated}): {current_state['quantity']:.8f} ETH à {exit_price_simulated:.2f} USDT.")
            value_at_entry = current_state['entry_price'] * current_state['quantity']
            commission_entry_abs = value_at_entry * COMMISSION_RATE
            value_at_exit = current_state['quantity'] * exit_price_simulated
            commission_exit_abs = value_at_exit * COMMISSION_RATE

            pnl_abs_trade = (value_at_exit - commission_exit_abs) - (value_at_entry + commission_entry_abs)
            pnl_pct_trade = pnl_abs_trade / value_at_entry if value_at_entry > 1e-9 else 0  # PnL % sur le capital investi

            current_state['paper_capital'] += pnl_abs_trade
            logging.info(
                f"Trade clôturé. PnL: {pnl_abs_trade:+.2f} USDT ({pnl_pct_trade:+.2%}). Nouveau Capital Papier: {current_state['paper_capital']:.2f} USDT")

            log_trade_to_csv({
                "entry_time_utc": current_state['entry_time_utc'].isoformat() if current_state[
                    'entry_time_utc'] else None,
                "entry_price": current_state['entry_price'], "exit_time_utc": current_time_utc.isoformat(),
                "exit_price": exit_price_simulated, "quantity": current_state['quantity'],
                "pnl_abs": pnl_abs_trade, "pnl_pct": pnl_pct_trade, "exit_reason": exit_reason_simulated,
                "commission_total": commission_entry_abs + commission_exit_abs,
                "capital_before_trade": capital_before_action,  # Capital avant PnL du trade
                "capital_after_trade": current_state['paper_capital']})
            current_state['in_position'] = False;
            current_state['entry_price'] = 0.0;
            current_state['quantity'] = 0.0
            current_state['sl_price'] = 0.0;
            current_state['tp_price'] = 0.0;
            current_state['entry_time_utc'] = None

    if not current_state['in_position'] and final_signal == 1:
        sl_price_calc = current_price * (1 - BEST_PARAMS['sl'])
        risk_per_unit = current_price - sl_price_calc
        if risk_per_unit > 1e-8:
            capital_to_risk_on_trade = current_state['paper_capital'] * RISK_PER_TRADE_PCT
            quantity_to_buy = capital_to_risk_on_trade / risk_per_unit
            value_of_trade = quantity_to_buy * current_price
            commission_entry_abs = value_of_trade * COMMISSION_RATE

            if value_of_trade + commission_entry_abs > current_state['paper_capital'] * 0.98:
                quantity_to_buy = (current_state['paper_capital'] * 0.98) / (current_price * (1 + COMMISSION_RATE))
                logging.warning(f"Quantité ajustée: {quantity_to_buy:.8f}")
                value_of_trade = quantity_to_buy * current_price
                commission_entry_abs = value_of_trade * COMMISSION_RATE

            if quantity_to_buy > 0:
                current_state['paper_capital'] -= commission_entry_abs  # Déduire frais d'entrée
                current_state['in_position'] = True;
                current_state['entry_price'] = current_price
                current_state['quantity'] = quantity_to_buy;
                current_state['sl_price'] = sl_price_calc
                current_state['tp_price'] = current_price * (1 + BEST_PARAMS['tp']);
                current_state['entry_time_utc'] = current_time_utc
                logging.info(
                    f"ORDRE D'ACHAT SIMULÉ: {quantity_to_buy:.8f} ETH à {current_price:.2f} USDT. SL: {sl_price_calc:.2f}, TP: {current_state['tp_price']:.2f}. Capital après frais: {current_state['paper_capital']:.2f}")
            else:
                logging.info("Signal d'achat mais quantité nulle. Pas de trade.")
        else:
            logging.warning(f"Risque par unité nul/négatif à {current_time_utc}. Pas de trade.")

    current_state['last_known_equity'] = current_state['paper_capital']
    save_state(current_state)
    logging.info(f"--- Fin Itération Bot. Capital Papier: {current_state['paper_capital']:.2f} USDT ---")


# --- Boucle principale de planification ---
if __name__ == '__main__':
    if API_KEY == "VOTRE_CLE_API_KUCOIN":
        logging.error("Clés API KuCoin non configurées. Veuillez les définir.");
        exit()

    logging.info(f"Démarrage du Bot de Paper Trading: {STRATEGY_NAME}")
    if not os.path.exists(TRADES_LOG_FILE):
        with open(TRADES_LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['EntryTimeUTC', 'EntryPrice', 'ExitTimeUTC', 'ExitPrice', 'Quantity', 'PnL_Abs', 'PnL_Pct',
                             'ExitReason', 'CommissionTotal', 'CapitalAvant', 'CapitalApres'])
    try:
        trading_bot_logic()
    except Exception as e:
        logging.exception("Erreur lors de la première exécution de trading_bot_logic")

    schedule.every().hour.at(":01").do(trading_bot_logic)
    # schedule.every(1).minutes.do(trading_bot_logic) # Pour test rapide

    logging.info("Planificateur démarré. Prochaine vérification à H:01.")
    while True:
        schedule.run_pending()
        time.sleep(1)
