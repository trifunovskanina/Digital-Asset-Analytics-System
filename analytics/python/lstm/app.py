from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from lstm.lstm import (
    LOOKBACK,
    load_ohlcv,
    prepare_features,
    create_lags,
    split_data,
    LSTMRegressor,
    train_and_evaluate,
    predict_future_prices,
    evaluate
)

# python -m uvicorn app:app --port 8000
# localhost:8000/predict
# localhost:8000/tech-analysis
app = FastAPI()

class LstmRequest(BaseModel):
    symbol: str
    n_future_days: int = 10

@app.post("/predict")
def predict(req: LstmRequest):
    symbol = req.symbol
    n_future_days = req.n_future_days

    EPOCHS = 200
    BATCH_SIZE = 32

    df = load_ohlcv(symbol)
    df = prepare_features(df)

    # not enough data case
    if len(df) <= LOOKBACK + 10:
        return {
            "symbol": symbol,
            "lookback": LOOKBACK,
            "metrics": None,
            "validation": {"true": [], "pred": []},
            "future": [],
        }

    df = create_lags(df)

    (train_X, val_X, test_X,
     train_y, val_y, test_y, scaler_x, scaler_y) = split_data(df)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_X.shape[2]

    model = LSTMRegressor(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)

    test_loss, true_y, pred_y = evaluate(model, test_loader, criterion)

    true_y = scaler_y.inverse_transform(true_y.cpu().numpy()).flatten()
    pred_y = scaler_y.inverse_transform(pred_y.cpu().numpy()).flatten()

    rmse = float(np.sqrt(mean_squared_error(true_y, pred_y)))
    mape = float(mean_absolute_percentage_error(true_y, pred_y))
    r2 = float(r2_score(true_y, pred_y))

    last_closes = df["close"].values[-LOOKBACK:]

    if len(last_closes) < LOOKBACK:
        return {
            "symbol": symbol,
            "lookback": LOOKBACK,
            "metrics": None,
            "validation": {"true": [], "pred": []},
            "future": [],
        }

    future_prices = predict_future_prices(
        model,
        last_closes,
        scaler_x,
        scaler_y,
        n_future_days,
    )

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        periods=n_future_days + 1,
        freq="D",
    )[1:]

    future_list = [
        {
            "date": dt.strftime("%Y-%m-%d"),
            "predicted_close": float(price),
        }
        for dt, price in zip(future_dates, future_prices)
    ]

    return {
        "symbol": symbol,
        "lookback": LOOKBACK,
        "metrics": {
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
        },
        "validation": {
            "true": true_y.tolist(),
            "pred": pred_y.tolist(),
        },
        "future": future_list,
    }