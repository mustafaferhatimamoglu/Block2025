import pandas as pd
import numpy as np
from typing import List, Tuple
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


def load_csv(path: str = "ohlc.csv") -> pd.DataFrame:
    """Read OHLCV data from a CSV file."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, EMA, MACD and Bollinger Bands columns."""
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


def label_data(
    df: pd.DataFrame, horizon: int = 4, threshold: float = 0.02
) -> pd.DataFrame:
    """Label rows as BUY (1) if price rises ``threshold`` in ``horizon`` hours."""
    df = df.copy()
    future = df["close"].shift(-horizon)
    pct_change = future / df["close"] - 1.0
    df["target"] = (pct_change >= threshold).astype(int)
    df.dropna(inplace=True)
    return df


def create_sequences(
    df: pd.DataFrame,
    features: List[str],
    lookback: int = 48,
    horizon: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Return sliding window sequences and labels."""
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


def create_dataset(
    df: pd.DataFrame,
    days: int = 30,
    lookback: int = 48,
    horizon: int = 4,
    threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """Prepare model input/output arrays from the last ``days`` of data."""
    df = df.tail(days * 24 + lookback + horizon)
    df = add_indicators(df)
    df = label_data(df, horizon=horizon, threshold=threshold)
    feature_cols = [
        "close",
        "volume",
        "rsi",
        "ema_20",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_mavg",
        "bb_high",
        "bb_low",
    ]
    X, y, idx = create_sequences(df, feature_cols, lookback, horizon)
    return X, y, idx, feature_cols


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
    horizon: int = 4,
    initial_balance: float = 1000.0,
    fee: float = 0.002,
) -> Tuple[float, List[dict]]:
    """Simulate simple trading and return final balance and trade log."""
    balance = initial_balance
    holdings = 0.0
    trades: List[dict] = []
    X, _, indices = create_sequences(df, features, lookback, horizon)
    preds = predict(model, X)
    closes = df["close"].iloc[lookback : len(df) - horizon + 1]
    for ts, pred, price in zip(indices, preds, closes):
        if pred == 1 and balance > 0:
            qty = (balance * (1 - fee)) / price
            holdings += qty
            trades.append({"time": ts, "type": "BUY", "price": price, "qty": qty})
            balance = 0.0
        elif pred == 0 and holdings > 0:
            proceeds = holdings * price * (1 - fee)
            balance += proceeds
            trades.append({"time": ts, "type": "SELL", "price": price, "qty": holdings})
            holdings = 0.0
    if holdings > 0:
        final_price = df["close"].iloc[-1]
        proceeds = holdings * final_price * (1 - fee)
        balance += proceeds
        trades.append({"time": df.index[-1], "type": "SELL", "price": final_price, "qty": holdings})
        holdings = 0.0
    return balance, trades


def main() -> None:
    df = load_csv()
    X, y, idx, features = create_dataset(df)
    model = train_model(X, y)
    final_balance, trades = simulate_trading(df, model, features)
    for t in trades:
        print(f"{t['time']}: {t['type']} {t['qty']:.4f} at {t['price']:.4f}")
    print(f"\nFinal balance: {final_balance:.2f} USDT")


if __name__ == "__main__":
    main()


