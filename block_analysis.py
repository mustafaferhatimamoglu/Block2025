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

from request_utils import get_with_retry
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
                df = pd.read_csv(
                    self.cache_file, parse_dates=["date"], index_col="date"
                )
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
            try:
                resp = get_with_retry(url, params=params, headers=headers, timeout=10)
            except requests.RequestException as exc:
                print(f"Error fetching data from CoinGecko: {exc}")
                break
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
        df["Bollinger_Upper"], df["Bollinger_Lower"] = self._bollinger(
            df["price"], window=20
        )
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

    def __init__(
        self, df: pd.DataFrame, seq_len: int = SEQ_LEN, epochs: int = EPOCHS
    ) -> None:
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

    def __init__(
        self, initial_balance: float = 100.0, commission: float = 0.002
    ) -> None:
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

    def run_daily(self) -> List[SimulationResult]:
        """Return daily balances by spreading monthly changes evenly."""
        months = [
            ("March", 31, 0.05),
            ("April", 30, -0.03),
            ("May", 31, 0.04),
            ("June", 30, 0.02),
            ("July", 31, -0.01),
            ("August", 31, 0.03),
            ("September", 30, -0.02),
            ("October", 31, 0.04),
        ]
        balance = self.initial_balance
        history: List[SimulationResult] = []
        for month, days, change in months:
            daily_rate = (1 + change) ** (1 / days) - 1
            for day in range(1, days + 1):
                balance -= balance * self.commission
                balance *= 1 + daily_rate
                balance -= balance * self.commission
                label = f"{month} {day}"
                history.append(SimulationResult(label, balance))
        return history


class LSTMTrader:
    """Backtest a naive strategy using LSTM price predictions.

    This example is provided for informational and educational purposes only and
    does not constitute financial advice.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = SEQ_LEN,
        train_size: int = 300,
        epochs: int = EPOCHS,
        initial_balance: float = 1000.0,
        commission: float = 0.002,
    ) -> None:
        self.df = df
        self.seq_len = seq_len
        self.train_size = train_size
        self.epochs = epochs
        self.initial_balance = initial_balance
        self.commission = commission
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model((seq_len, 1))
        self.trades: List[dict] = []

    def _build_model(self, input_shape) -> Sequential:
        model = Sequential(
            [Input(shape=input_shape), LSTM(50, activation="relu"), Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def _prepare_data(self) -> np.ndarray:
        return self.scaler.fit_transform(self.df[["price"]])

    def backtest(self) -> float:
        scaled = self._prepare_data()

        X_train, y_train = [], []
        for i in range(self.train_size - self.seq_len):
            X_train.append(scaled[i : i + self.seq_len])
            y_train.append(scaled[i + self.seq_len])
        self.model.fit(
            np.array(X_train), np.array(y_train), epochs=self.epochs, verbose=0
        )

        balance = self.initial_balance
        holdings = 0.0
        for i in range(self.train_size, len(self.df) - 1):
            seq = scaled[i - self.seq_len : i]
            pred_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            price = self.df["price"].iloc[i]

            if pred > price and balance > 0:
                qty = balance / price
                balance -= qty * price * (1 + self.commission)
                holdings += qty
                self.trades.append(
                    {
                        "time": self.df.index[i],
                        "type": "BUY",
                        "price": price,
                        "qty": qty,
                    }
                )
            elif pred < price and holdings > 0:
                balance += holdings * price * (1 - self.commission)
                self.trades.append(
                    {
                        "time": self.df.index[i],
                        "type": "SELL",
                        "price": price,
                        "qty": holdings,
                    }
                )
                holdings = 0.0

        if holdings > 0:
            final_price = self.df["price"].iloc[-1]
            balance += holdings * final_price * (1 - self.commission)
            self.trades.append(
                {
                    "time": self.df.index[-1],
                    "type": "SELL",
                    "price": final_price,
                    "qty": holdings,
                }
            )
        return balance


def plot_predictions(
    df: pd.DataFrame, predictions: pd.Series, outfile: str = "prediction.png"
) -> None:
    """Plot actual prices and predicted future prices."""
    last_100 = df[-100:]
    ax = last_100["price"].plot(
        title="Blockasset (BLOCK) Price Prediction", label="Actual"
    )
    pred_series = pd.concat(
        [
            pd.Series({last_100.index[-1]: last_100["price"].iloc[-1]}),
            predictions,
        ]
    )
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

    # Show an example of daily balances
    daily_results = simulator.run_daily()
    print("\nFirst 7 daily balances:")
    for res in daily_results[:7]:
        print(f"{res.month}: ${res.balance:.2f}")

    # Demonstrate LSTM-based trading
    trader = LSTMTrader(raw_df)
    final_balance = trader.backtest()
    print(f"\nLSTM trading simulation final balance: ${final_balance:.2f}")
    if trader.trades:
        print("First 3 trades:")
        for t in trader.trades[:3]:
            print(
                f"{t['time']:%Y-%m-%d %H:%M} {t['type']} at ${t['price']:.4f} qty {t['qty']:.4f}"
            )


if __name__ == "__main__":
    main()
