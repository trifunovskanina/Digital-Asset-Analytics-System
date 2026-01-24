import sqlite3
import pandas as pd
import numpy as np
import json
import sys
import os
import warnings
import psycopg2

warnings.filterwarnings("ignore")

TIMEFRAME_RULES = {
    "1d": "D",
    "1w": "W",
    "1m": "M",
}

def load_ohlcv_tech(symbol: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host="postgres",
        database="analytics",
        user="postgres",
        password="admin",
        port=5432
    )

    query = """
        SELECT date, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s
        ORDER BY date;
    """

    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df



def resample_ohlcv(df, rule):

    ohlc = df[["open", "high", "low", "close"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )

    vol = df["volume"].resample(rule).sum()

    out = ohlc.copy()
    out["volume"] = vol

    return out.dropna()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)

    def calc(x):
        return np.dot(x, weights) / weights.sum()

    return series.rolling(period).apply(calc, raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1 / period, adjust=False).mean()

    rs = gain_ema / loss_ema
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return macd_line, signal_line, hist


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()

    return k, d


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    plus_dm = high.diff()
    minus_dm_raw = low.diff()

    # directional movement
    plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm_raw.abs()), plus_dm, 0.0)
    minus_dm = np.where((minus_dm_raw < 0) & (minus_dm_raw.abs() > plus_dm), minus_dm_raw.abs(), 0.0)

    tr = true_range(df)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=period).sum() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()

    return adx_val


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0  # typical price
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_val = (tp - sma_tp) / (0.015 * mad)

    return cci_val


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def volume_ma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Oscillators
    out["rsi14"] = rsi(out["close"], 6)
    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    k, d = stochastic_oscillator(out)
    out["stoch_k"] = k
    out["stoch_d"] = d
    out["adx14"] = adx(out, 6)
    out["cci20"] = cci(out, 6)

    # Moving averages
    out["sma20"] = sma(out["close"], 6)
    out["ema20"] = ema(out["close"], 6)
    out["wma20"] = wma(out["close"], 6)

    bb_mid, bb_upper, bb_lower = bollinger_bands(out["close"], 6, 2.0)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    out["vma20"] = volume_ma(out["volume"], 6)

    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sig_rsi"] = np.where(out["rsi14"] < 30, "buy",
                              np.where(out["rsi14"] > 70, "sell", "hold"))

    out["sig_macd"] = "hold"
    macd_above = out["macd"] > out["macd_signal"]
    macd_cross_up = macd_above & (~macd_above.shift(1).fillna(False))
    macd_cross_down = (~macd_above) & (macd_above.shift(1).fillna(False))
    out.loc[macd_cross_up, "sig_macd"] = "buy"
    out.loc[macd_cross_down, "sig_macd"] = "sell"

    out["sig_stoch"] = np.where(out["stoch_k"] < 20, "buy",
                                np.where(out["stoch_k"] > 80, "sell", "hold"))

    out["sig_cci"] = np.where(out["cci20"] < -100, "buy",
                              np.where(out["cci20"] > 100, "sell", "hold"))

    strong_trend = out["adx14"] > 25
    above_sma = out["close"] > out["sma20"]
    out["sig_adx"] = "hold"
    out.loc[strong_trend & above_sma, "sig_adx"] = "buy"
    out.loc[strong_trend & (~above_sma), "sig_adx"] = "sell"

    out["sig_sma"] = np.where(out["close"] > out["sma20"], "buy",
                              np.where(out["close"] < out["sma20"], "sell", "hold"))

    out["sig_ema"] = np.where(out["close"] > out["ema20"], "buy",
                              np.where(out["close"] < out["ema20"], "sell", "hold"))

    out["sig_wma"] = np.where(out["close"] > out["wma20"], "buy",
                              np.where(out["close"] < out["wma20"], "sell", "hold"))

    out["sig_bb"] = np.where(out["close"] < out["bb_lower"], "buy",
                             np.where(out["close"] > out["bb_upper"], "sell", "hold"))

    out["sig_vma"] = np.where(
        (out["close"] > out["sma20"]) & (out["volume"] > out["vma20"]), "buy",
        np.where((out["close"] < out["sma20"]) & (out["volume"] > out["vma20"]), "sell", "hold")
    )

    sig_cols = ["sig_rsi", "sig_macd", "sig_stoch", "sig_adx", "sig_cci",
                "sig_sma", "sig_ema", "sig_wma", "sig_bb", "sig_vma"]

    def majority_vote(row):
        counts = row.value_counts()
        return counts.idxmax()

    out["signal_overall"] = out[sig_cols].apply(majority_vote, axis=1)

    return out


def compute(symbol: str, timeframe: str = "1d", limit: int = 100):
    df_daily = load_ohlcv(DB_PATH, symbol)

    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rule = TIMEFRAME_RULES[timeframe]
    if rule == "D":
        df = df_daily
    else:
        df = resample_ohlcv(df_daily, rule)

    df_ind = add_all_indicators(df)
    df_full = generate_signals(df_ind)

    return df_full.tail(limit)


def main():
    # python tech_analysis.py <symbol> <timeframe> <limit>
    symbol = sys.argv[1] if len(sys.argv) > 1 else "bitcoin"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    df = compute(symbol, timeframe, limit)

    # reset index so 'date' becomes a normal column, then dump as JSON
    df = df.reset_index()
    data = df.to_dict(orient="records")
    print(json.dumps(data, default=str))  # default=str for datetime


if __name__ == "__main__":
    main()
