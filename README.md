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

Use `fetch_ohlc.py` to download open, high, low and close prices for the
Blockasset token from CoinGecko. By default the script retrieves the last
365 days of hourly data. The values are saved to a CSV file for later
inspection. You can specify the output file with `--outfile` and adjust the
history length with `--days` (maximum 365).

```bash
python fetch_ohlc.py --outfile my_data.csv --days 120

```

The script prints the first few rows of data and reports how many entries
were written to the file.
