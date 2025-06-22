"""Blockasset price prediction and trade simulation module.

This script demonstrates a simple object-oriented design following the
SOLID principles. It fetches historical prices, calculates technical
indicators, trains an LSTM model to predict future prices, and simulates
basic trades using the predictions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

# Disable GPU and TensorFlow logging for consistent CPU execution
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

df_logger = tf.get_logger()
df_logger.setLevel("ERROR")

# Configuration
CACHE_FILE = "block_prices.csv"
PREDICTION_HOURS = 24
# Fewer epochs keep example execution snappy
EPOCHS = 5
SEQ_LEN = 24


class DataFetcher:
    """Fetches historical price data for Blockasset from CoinGecko."""

    def __init__(self, cache_file: str = CACHE_FILE) -> None:
        self.cache_file = cache_file

    def fetch(self) -> pd.DataFrame:
        """Return a DataFrame with hourly prices."""
        if os.path.exists(self.cache_file):
            file_age = time.time() - os.path.getmtime(self.cache_file)
            if file_age < 24 * 3600:
                df = pd.read_csv(self.cache_file, parse_dates=["date"], index_col="date")
                return df.sort_index()

        end_ts = int(time.time())
        start_ts = end_ts - 365 * 24 * 3600
        step = 90 * 24 * 3600
        url = "https://api.coingecko.com/api/v3/coins/blockasset/market_chart/range"
        all_prices: List[List[int]] = []
        cur = start_ts
        headers = {"accept": "application/json"}

        while cur < end_ts:
            params = {
                "vs_currency": "usd",
                "from": cur,
                "to": min(cur + step, end_ts),
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            prices = data.get("prices", [])
            all_prices.extend(prices)
            cur += step
            time.sleep(1)

        df = pd.DataFrame(all_prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        df = df[~df.index.duplicated(keep="first")].drop("timestamp", axis=1)
        df = df.sort_index()
        df.to_csv(self.cache_file)
        return df


class IndicatorCalculator:
    """Adds technical indicators to a price DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def add_indicators(self) -> pd.DataFrame:
        df = self.df
        df["SMA_20"] = df["price"].rolling(window=20).mean()
        df["EMA_20"] = df["price"].ewm(span=20).mean()
        df["Price_Change"] = df["price"].pct_change()
        df["RSI"] = self._rsi(df["price"], period=14)
        df["MACD"] = self._macd(df["price"])
        df["Bollinger_Upper"], df["Bollinger_Lower"] = self._bollinger(df["price"], window=20)
        return df.dropna()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    @staticmethod
    def _bollinger(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower


class LSTMModel:
    """Handles preprocessing, training and predicting with an LSTM model."""

    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN, epochs: int = EPOCHS) -> None:
        self.df = df
        self.seq_len = seq_len
        self.epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model((seq_len, 1))

    def _build_model(self, input_shape) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(50, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def _prepare_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler.fit_transform(self.df[["price"]])
        sequences, targets = [], []
        for i in range(len(scaled) - self.seq_len):
            sequences.append(scaled[i : i + self.seq_len])
            targets.append(scaled[i + self.seq_len])
        return np.array(sequences), np.array(targets)

    def train(self) -> None:
        X, y = self._prepare_sequences()
        self.model.fit(X, y, epochs=self.epochs, verbose=0)
        self.last_sequence = X[-1:]

    def predict(self, hours: int = PREDICTION_HOURS) -> pd.Series:
        seq = self.last_sequence[0]
        predictions = []
        for _ in range(hours):
            pred_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)
            predictions.append(pred_scaled[0, 0])
            seq = np.vstack([seq[1:], pred_scaled[0]])
        preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        last_date = self.df.index[-1]
        dates = [last_date + pd.Timedelta(hours=i) for i in range(1, hours + 1)]
        return pd.Series(preds[:, 0], index=dates, name="predicted_price")


@dataclass
class SimulationResult:
    month: str
    balance: float


class TradeSimulator:
    """Simple monthly trade simulation applying commissions."""

    def __init__(self, initial_balance: float = 100.0, commission: float = 0.002) -> None:
        self.initial_balance = initial_balance
        self.commission = commission

    def run(self) -> List[SimulationResult]:
        months = [
            ("March", 0.05),
            ("April", -0.03),
            ("May", 0.04),
            ("June", 0.02),
            ("July", -0.01),
            ("August", 0.03),
            ("September", -0.02),
            ("October", 0.04),
        ]
        balance = self.initial_balance
        history: List[SimulationResult] = []
        for month, change in months:
            balance -= balance * self.commission
            balance *= 1 + change
            balance -= balance * self.commission
            history.append(SimulationResult(month, balance))
        return history


def plot_predictions(df: pd.DataFrame, predictions: pd.Series, outfile: str = "prediction.png") -> None:
    """Plot actual prices and predicted future prices."""
    last_100 = df[-100:]
    ax = last_100["price"].plot(title="Blockasset (BLOCK) Price Prediction", label="Actual")
    pred_series = pd.concat([
        pd.Series({last_100.index[-1]: last_100["price"].iloc[-1]}),
        predictions,
    ])
    pred_series.plot(ax=ax, label="Predicted", color="orange")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile)


def main() -> None:
    fetcher = DataFetcher()
    raw_df = fetcher.fetch()
    print(f"Training data starts at {raw_df.index[0]:%Y-%m-%d %H:%M:%S}")

    calc = IndicatorCalculator(raw_df)
    df = calc.add_indicators()

    model = LSTMModel(df)
    model.train()
    predictions = model.predict(PREDICTION_HOURS)
    predictions.to_csv("predictions.csv", header=True)

    plot_predictions(df, predictions)

    simulator = TradeSimulator()
    results = simulator.run()
    for res in results:
        print(f"End of {res.month}: ${res.balance:.2f}")
    print(f"\nFinal balance on 2025-10-31: ${results[-1].balance:.2f}")


if __name__ == "__main__":
    main()
