import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Fetch daily price data for Blockasset (BLOCK) from CoinGecko

def fetch_data():
    url = "https://api.coingecko.com/api/v3/coins/blockasset/market_chart"
    params = {
        "vs_currency": "usd",
        # using a fixed window as CoinGecko may reject very large ranges
        "days": 365,
        "interval": "daily",
    }
    headers = {"accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching data from CoinGecko: {exc}")
        return pd.DataFrame()
    data = resp.json()
    # Extract timestamps and prices
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Preprocess data: scale prices and create sequences

def preprocess_data(df, seq_len=5):
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
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the LSTM model

def train_model(model, X_train, y_train, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

# Predict the next day price using the last sequence

def predict_next_day(model, last_sequence, scaler):
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)
    return pred_price[0, 0]

if __name__ == "__main__":
    df = fetch_data()
    X, y, scaler = preprocess_data(df)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model = train_model(model, X, y, epochs=50)

    # Use the last seq_len days to predict the next day
    last_sequence = X[-1:]
    next_day_price = predict_next_day(model, last_sequence, scaler)
    print(f"Predicted next day price: ${next_day_price:.4f}")

    # Plot the last 100 actual prices and the prediction
    last_100 = df[-100:].copy()
    last_100['Predicted'] = np.nan
    last_100.iloc[-1, last_100.columns.get_loc('Predicted')] = next_day_price
    last_100[['price', 'Predicted']].plot(title="Blockasset (BLOCK) Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()
