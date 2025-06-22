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

```
python block_price_prediction.py
```

The chart will be saved to `prediction.png` in the current directory. Use the
`--show` flag if you want to display the plot as well. You can also control the
number of training epochs with the `--epochs` option:

```
python block_price_prediction.py --show --epochs 5

```

You can adjust how many hours to predict with the `--hours` option. For
example, to predict the next 12 hours:

```
python block_price_prediction.py --hours 12
```

## Trade Simulation

A simple script `trade_simulation.py` shows a hypothetical trading example. It starts with $100 on 2025-03-01 and applies a 0.2% commission to each buy and sell. Run it with:

```bash
python trade_simulation.py
```
do not change 
@
.venv/bin/pip install -r requirements.txt 
.venv/bin/python block_price_prediction.py 
.venv/bin/python trade_simulation.py
@

