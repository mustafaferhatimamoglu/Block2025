# Block2025

This repository contains a simple script to predict the Blockasset (BLOCK) price using an LSTM neural network. The script fetches historical price data from CoinGecko and trains a model using the CPU-only TensorFlow package and scikit-learn.

## Installation

Install the Python dependencies (it's recommended to do this inside a virtual environment). The requirements use the CPU-only TensorFlow build, which avoids CUDA-related errors:

```
pip install -r requirements.txt
```
The requirements file uses `tensorflow-cpu`, which installs TensorFlow without any GPU libraries.

Note that this project uses the CPU-only TensorFlow package (`tensorflow-cpu`). Installing the GPU variant (`tensorflow`) on a system without CUDA may produce errors.

**Note:** The script is designed to run on the CPU. If your environment
lacks GPU support, install the `tensorflow-cpu` package to avoid any
CUDA-related errors.

## Usage

Run the prediction script:

The script downloads the latest price data from CoinGecko, trains an LSTM model
and prints the predicted prices for the next 24 hours by default. The fetched
prices are cached in `block_prices.csv`; if this file is newer than 24 hours the
data is loaded from disk instead of downloading again. After training a chart
window will open showing recent prices along with the prediction.

```bash
python block_analysis.py
```

The script computes several technical indicators, trains an LSTM model and
predicts the next 24 hours of prices. Results are written to `predictions.csv`
and `prediction.png`. A small trade simulation is executed automatically and
printed to the console.

## Trade Simulation

The previous standalone `trade_simulation.py` has been incorporated into
`block_analysis.py`. When you run the analysis script a short example simulation
is executed automatically and the monthly balances are printed.


## Fetching OHLC Data

Use `fetch_ohlc.py` to download open, high, low and close prices from CoinGecko. The script prints the first few rows or saves them to CSV.

```bash
python fetch_ohlc.py bitcoin --days 30 --outfile btc_ohlc.csv
```

The `days` option accepts `1, 7, 14, 30, 90, 180, 365` or `max`.
