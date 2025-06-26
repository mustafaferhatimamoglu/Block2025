# Block2025

This repository contains a simple script to predict the Blockasset (BLOCK) price using an LSTM neural network. The script fetches historical price data from CoinGecko and trains a model using the CPU-only TensorFlow package and scikit-learn.

## Installation

Install the Python dependencies (it's recommended to do this inside a virtual environment). The requirements use the CPU-only TensorFlow build, which avoids CUDA-related errors. A new dependency, `joblib`, is included for saving and loading models:

```
pip install -r requirements.txt
```
The requirements file uses `tensorflow-cpu`, which installs TensorFlow without any GPU libraries, and also lists `joblib` for saving and loading models.

Note that this project uses the CPU-only TensorFlow package (`tensorflow-cpu`). Installing the GPU variant (`tensorflow`) on a system without CUDA may produce errors.

**Note:** The script is designed to run on the CPU. If your environment
lacks GPU support, install the `tensorflow-cpu` package to avoid any
CUDA-related errors.

## Usage

Run the prediction script:

The script downloads the latest price data from CoinGecko, trains an LSTM model
and prints the predicted prices for the next 24 hours by default. The fetched
prices are cached in `block_prices.csv`; if this file is newer than 24 hours the
data is loaded from disk instead of downloading again. Should the API respond
with "Too Many Requests" errors, the fetcher now retries automatically using a
simple exponential backoff. After training a chart window will open showing
recent prices along with the prediction.


```bash
python block_analysis.py
```

You can also run the analysis on your own Gate.io CSV export using the
`--csv` option. The loader expects a column named `close`, which will be
treated as the price.

```bash
python block_analysis.py --csv my_data.csv
```

The script computes several technical indicators, trains an LSTM model and
predicts the next 24 hours of prices. Results are written to `predictions.csv`
and `prediction.png`. A small trade simulation is executed automatically and
printed to the console.

## Trade Simulation

The previous standalone `trade_simulation.py` has been incorporated into
`block_analysis.py`. When you run the analysis script a short example simulation
is executed automatically and the monthly balances are printed.

The `TradeSimulator` also provides a `run_daily()` method that spreads the
monthly returns over each day and returns a list of daily balances. In the
example script the first few daily results are printed after the monthly
summary.

## LSTM Trading Example

`block_analysis.py` also demonstrates an experimental `LSTMTrader` that trains
on historical prices and performs a very naive backtest. After the monthly
simulation finishes, the script runs the trader and prints the final balance and
some sample trades. Charts of the trades and account balance are saved to
`trades.png` and `balance.png` for quick inspection. The model now trains for
20 epochs by default which can yield slightly more stable results.

## Disclaimer

The trading examples in this repository are for informational and educational
purposes only. They do not constitute financial advice. Use any strategy at your
own risk.


## Fetching OHLC Data

Use `fetch_ohlc.py` to download hourly candlestick data for the Blockasset
token from gate.io. All values returned by the API—timestamp, open, high, low,
close, volume, quote volume and the completion flag—are written to a CSV file.
By default the script retrieves the last 365 days of history. You can specify
the output file with `--outfile` and adjust the history length with `--days`
(maximum 365).

```bash
python fetch_ohlc.py --outfile my_data.csv --days 120
```

The script prints the first few rows of data and reports how many entries
were written to the file.

Once you have a CSV file you can run the main analysis script with:

```bash
python block_analysis.py --csv my_data.csv
```
