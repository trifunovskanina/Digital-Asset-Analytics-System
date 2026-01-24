from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import math

from tech.tech_analysis import (
    TIMEFRAME_RULES,
    load_ohlcv_tech,
    resample_ohlcv,
    add_all_indicators,
    generate_signals
)

app = FastAPI()

class TechRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    limit: int = 100

def _clean(v):
    # convert numpy scalars to python scalars
    if isinstance(v, (np.generic,)):
        v = v.item()

    # pandas timestamps to string
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d %H:%M:%S")

    # NaN / inf to None
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None

    # pandas NaT to numpy nan
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    return v

@app.post("/tech-analysis")
def tech_analysis(req: TechRequest):
    symbol = req.symbol.lower()
    timeframe = req.timeframe
    limit = req.limit

    if timeframe not in TIMEFRAME_RULES:
        return {"symbol": symbol, "timeframe": timeframe, "data": []}

    df_daily = load_ohlcv_tech(symbol)
    if df_daily.empty:
        return {"symbol": symbol, "timeframe": timeframe, "data": []}

    rule = TIMEFRAME_RULES[timeframe]
    df = df_daily if rule == "D" else resample_ohlcv(df_daily, rule)

    min_len = 30 if timeframe != "1m" else 12
    if len(df) < min_len:
        return {"symbol": symbol, "timeframe": timeframe, "data": []}

    df_ind = add_all_indicators(df)
    df_full = generate_signals(df_ind)

    df_out = df_full.tail(limit).reset_index()

    if "date" in df_out.columns:
        df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    records = df_out.to_dict(orient="records")
    records = [
        {k: _clean(v) for k, v in row.items()}
        for row in records
    ]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "data": records
    }
