import pandas as pd
import numpy as np
from typing import List, Tuple
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to the dataframe."""
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df


def label_data(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Label data as 1 (BUY) if future price is higher, else 0 (SELL)."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = np.where(df["future_close"] > df["close"], 1, 0)
    df.dropna(inplace=True)
    return df


def create_sequences(
    df: pd.DataFrame, features: List[str], lookback: int = 48, horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Create feature/label sequences for model training."""
    sequences: List[np.ndarray] = []
    targets: List[int] = []
    indices: List[pd.Timestamp] = []
    for i in range(lookback, len(df) - horizon + 1):
        seq = df[features].iloc[i - lookback : i].values
        label = df["target"].iloc[i]
        sequences.append(seq)
        targets.append(label)
        indices.append(df.index[i])
    return np.array(sequences), np.array(targets), indices


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForest classifier and return the trained model."""
    X_flat = X.reshape((X.shape[0], -1))
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestClassifier, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> RandomForestClassifier:
    return joblib.load(path)


def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    X_flat = X.reshape((X.shape[0], -1))
    return model.predict(X_flat)


def simulate_trading(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    features: List[str],
    lookback: int = 48,
    horizon: int = 24,
    initial_balance: float = 1000.0,
) -> Tuple[float, List[dict]]:
    """Simulate trading based on model predictions."""
    balance = initial_balance
    holdings = 0.0
    trades: List[dict] = []
    X, _, indices = create_sequences(df, features, lookback, horizon)
    preds = predict(model, X)
    closes = df["close"].iloc[lookback : len(df) - horizon + 1]
    for ts, pred, price in zip(indices, preds, closes):
        if pred == 1 and balance > 0:  # BUY
            qty = balance / price
            holdings += qty
            balance = 0.0
            trades.append({"time": ts, "type": "BUY", "price": price, "qty": qty})
        elif pred == 0 and holdings > 0:  # SELL
            balance += holdings * price
            trades.append({"time": ts, "type": "SELL", "price": price, "qty": holdings})
            holdings = 0.0
    # liquidate remaining holdings at final price
    if holdings > 0:
        final_price = df["close"].iloc[-1]
        balance += holdings * final_price
        trades.append({"time": df.index[-1], "type": "SELL", "price": final_price, "qty": holdings})
        holdings = 0.0
    return balance, trades


