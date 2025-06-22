import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
import ta

# Suppress verbose TensorFlow logs and disable GPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")
tf.get_logger().setLevel("ERROR")


class DataFetcher:
    """Fetch hourly Blockasset price data with local caching."""

    def __init__(self, cache_file: str = "block_prices.csv") -> None:
        self.cache_file = cache_file

    def fetch(self) -> pd.DataFrame:
        """Return a DataFrame of prices and volumes from the last year."""
        if os.path.exists(self.cache_file):
            try:
                file_age = time.time() - os.path.getmtime(self.cache_file)
                if file_age < 24 * 3600:
                    df = pd.read_csv(self.cache_file, parse_dates=["date"], index_col="date")
                    return df.sort_index()
            except Exception as exc:  # pragma: no cover - fall back to network
                print(f"Error loading cached data: {exc}")
        df = self._download()
        df.to_csv(self.cache_file)
        return df

    def _download(self) -> pd.DataFrame:
        """Download data from CoinGecko."""
        end_ts = int(time.time())
        start_ts = end_ts - 365 * 24 * 3600
        step = 90 * 24 * 3600
        url = "https://api.coingecko.com/api/v3/coins/blockasset/market_chart/range"
        headers = {"accept": "application/json"}

        rows: List[dict[str, float]] = []
        cur = start_ts
        while cur < end_ts:
            params = {"vs_currency": "usd", "from": cur, "to": min(cur + step, end_ts)}
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"Error fetching data from CoinGecko: {exc}")
                break
            data = resp.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            for price, vol in zip(prices, volumes):
                rows.append({"timestamp": price[0], "price": price[1], "volume": vol[1]})
            cur += step
            time.sleep(1)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        df = df[~df.index.duplicated(keep="first")].drop("timestamp", axis=1)
        return df.sort_index()

class IndicatorCalculator:
    """Add several technical indicators to a DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def add_indicators(self) -> pd.DataFrame:
        df = self.df.copy()
        df["sma_14"] = ta.trend.sma_indicator(df["price"], window=14)
        df["ema_14"] = ta.trend.ema_indicator(df["price"], window=14)
        df["ema_50"] = ta.trend.ema_indicator(df["price"], window=50)
        df["rsi"] = ta.momentum.rsi(df["price"], window=14)
        df["macd"] = ta.trend.macd(df["price"])
        df["macd_signal"] = ta.trend.macd_signal(df["price"])
        bb = ta.volatility.BollingerBands(df["price"])
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mavg"] = bb.bollinger_mavg()
        if "volume" in df.columns:
            df["obv"] = ta.volume.on_balance_volume(df["price"], df["volume"])
        self.df = df
        return df


class PricePredictor:
    """Train an LSTM model and generate price predictions."""

    def __init__(self, epochs: int = 50, seq_len: int = 24) -> None:
        self.epochs = epochs
        self.seq_len = seq_len
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model: Sequential | None = None

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler.fit_transform(df[["price"]])
        sequences, targets = [], []
        for i in range(len(scaled) - self.seq_len):
            sequences.append(scaled[i : i + self.seq_len])
            targets.append(scaled[i + self.seq_len])
        return np.array(sequences), np.array(targets)

    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self._build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=self.epochs, verbose=0)

    def predict(self, last_sequence: np.ndarray, hours: int = 24) -> np.ndarray:
        seq = last_sequence[0]
        preds = []
        for _ in range(hours):
            pred_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)
            preds.append(pred_scaled[0, 0])
            seq = np.vstack([seq[1:], pred_scaled[0]])
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1))[:, 0]


class TradeSimulator:
    """Demonstrate a very simple monthly trading scenario."""

    def __init__(self, initial_balance: float = 100.0, commission: float = 0.002) -> None:
        self.initial_balance = initial_balance
        self.commission = commission

    def simulate(self) -> tuple[list[tuple[str, float]], pd.Timestamp, float]:
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
        history = []
        for month, change in months:
            balance -= balance * self.commission
            balance *= 1 + change
            balance -= balance * self.commission
            history.append((month, balance))
        end_date = pd.Timestamp(year=2025, month=10, day=31)
        return history, end_date, balance


class BlockPriceApp:
    """End-to-end application to fetch data, train model and simulate trades."""

    def __init__(self, epochs: int = 50, predict_hours: int = 24, show_plot: bool = False) -> None:
        self.epochs = epochs
        self.predict_hours = predict_hours
        self.show_plot = show_plot

    def run(self) -> None:
        fetcher = DataFetcher()
        df = fetcher.fetch()
        calc = IndicatorCalculator(df)
        df = calc.add_indicators()
        df.to_csv(fetcher.cache_file)

        predictor = PricePredictor(epochs=self.epochs)
        X, y = predictor.preprocess(df)
        predictor.train(X, y)
        predictions = predictor.predict(X[-1:], hours=self.predict_hours)

        print(f"Next {self.predict_hours} hour predictions:")
        for i, price in enumerate(predictions, 1):
            print(f"Hour +{i}: ${price:.4f}")

        self._plot(df, predictions)

        simulator = TradeSimulator()
        history, end_date, final_balance = simulator.simulate()
        for month, bal in history:
            print(f"End of {month}: ${bal:.2f}")
        print(f"\nFinal balance on {end_date:%Y-%m-%d}: ${final_balance:.2f}")

    def _plot(self, df: pd.DataFrame, predictions: np.ndarray) -> None:
        last_100 = df[-100:].copy()
        ax = last_100["price"].plot(title="Blockasset (BLOCK) Price Prediction", label="Actual")
        pred_dates = [last_100.index[-1] + pd.Timedelta(hours=i) for i in range(1, self.predict_hours + 1)]
        pred_series = pd.Series([last_100["price"].iloc[-1]] + predictions.tolist(), index=[last_100.index[-1]] + pred_dates)
        pred_series.plot(ax=ax, label="Predicted", color="orange")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.savefig("prediction.png")
        if self.show_plot:
            plt.show()


def main() -> None:
    app = BlockPriceApp()
    app.run()


if __name__ == "__main__":
    main()
