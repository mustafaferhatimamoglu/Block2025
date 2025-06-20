import argparse
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # suppress INFO and WARNING logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # disable GPU detection
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

# Fetch hourly price data for Blockasset (BLOCK) from CoinGecko

# Fetch hourly price data for Blockasset (BLOCK) from CoinGecko

def fetch_data() -> pd.DataFrame:
    """Fetch hourly price data for Blockasset from the last year."""

    end_ts = int(time.time())
    start_ts = end_ts - 365 * 24 * 3600
    step = 90 * 24 * 3600  # 90 days
    url = "https://api.coingecko.com/api/v3/coins/blockasset/market_chart/range"

    all_prices: List[List[int]] = []
    headers = {"accept": "application/json"}

    cur = start_ts
    while cur < end_ts:
        params = {
            "vs_currency": "usd",
            "from": cur,
            "to": min(cur + step, end_ts),
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
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
    df = df[~df.index.duplicated(keep="first")]
    df.drop("timestamp", axis=1, inplace=True)
    return df.sort_index()

# Preprocess data: scale prices and create hourly sequences


def preprocess_data(df, seq_len=24):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['price']])
    sequences = []
    targets = []
    for i in range(len(scaled) - seq_len):
        sequences.append(scaled[i:i + seq_len])
        targets.append(scaled[i + seq_len])
    X = np.array(sequences)
    y = np.array(targets)
    return X, y, scaler

# Build a simple LSTM model


def build_lstm_model(input_shape):
    """Build a simple LSTM model without the Keras input shape warning."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the LSTM model


def train_model(model, X_train, y_train, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

# Predict the next n hours using the last sequence


def predict_next_hours(model, last_sequence, scaler, n_hours=24):
    """Iteratively predict the next `n_hours` prices."""
    seq = last_sequence[0]
    predictions = []
    for _ in range(n_hours):
        pred_scaled = model.predict(seq[np.newaxis, :, :], verbose=0)
        predictions.append(pred_scaled[0, 0])
        seq = np.vstack([seq[1:], pred_scaled[0]])
    preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return preds[:, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Blockasset prices using an LSTM model"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving it to disk",
    )
    args = parser.parse_args()

    df = fetch_data()
    start_date = df.index[0]
    print(f"Training data starts at {start_date:%Y-%m-%d %H:%M:%S}")
    # Show the date range of the fetched data
    if not df.empty:
        print(f"Data start: {df.index[0]}  Data end: {df.index[-1]}")
    X, y, scaler = preprocess_data(df)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model = train_model(model, X, y, epochs=50)

    # Use the last seq_len hours to predict the next 24 hours
    last_sequence = X[-1:]
    next_prices = predict_next_hours(model, last_sequence, scaler, n_hours=24)
    print("Next 24 hour predictions:")
    for i, price in enumerate(next_prices, start=1):
        print(f"Hour +{i}: ${price:.4f}")

    # Plot the last 100 actual prices
    last_100 = df[-100:].copy()
    ax = last_100['price'].plot(
        title="Blockasset (BLOCK) Price Prediction",
        label="Actual",
    )

    # Plot predicted next 24 hours as a line starting from the last actual point
    pred_dates = [last_100.index[-1] + pd.Timedelta(hours=i) for i in range(1, 25)]
    pred_values = next_prices
    pred_series = pd.Series(
        [last_100['price'].iloc[-1]] + pred_values.tolist(),
        index=[last_100.index[-1]] + pred_dates,
    )
    pred_series.plot(ax=ax, label="Predicted", color="orange")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.savefig("prediction.png")
    if args.show:
        plt.show()
