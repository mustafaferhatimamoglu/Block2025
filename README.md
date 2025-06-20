# Block2025

This repository contains a simple script to predict the Blockasset (BLOCK) price using an LSTM neural network. The script fetches historical price data from CoinGecko and trains a model using TensorFlow and scikit-learn.

## Installation

Install the Python dependencies (it's recommended to do this inside a virtual environment):

```
pip install -r requirements.txt
```

## Usage

Run the prediction script:

The script downloads the latest price data from CoinGecko, trains an LSTM model and prints the predicted next day price. After training a chart window will open showing recent prices along with the prediction.

```
python block_price_prediction.py
```

Example output:

```
Predicted next day price: $0.1234
```
