# Block2025

This repository contains a simple script to predict the Blockasset (BLOCK) price using an LSTM neural network. The script fetches historical price data from CoinGecko and trains a model using TensorFlow and scikit-learn.

## Installation

Install the Python dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the prediction script:

```
python block_price_prediction.py
```

The chart will be saved to `prediction.png` in the current directory. Use the
`--show` flag if you want to display the plot as well:

```
python block_price_prediction.py --show
```
