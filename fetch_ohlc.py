import requests
import pandas as pd


COIN_ID = "blockasset"
VS_CURRENCY = "usd"
DAYS = 30  # returns 1 hour intervals for up to 90 days


def fetch_ohlc() -> pd.DataFrame:
    """Download OHLC values for the configured coin."""
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/ohlc"
    params = {"vs_currency": VS_CURRENCY, "days": DAYS}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    return df


def main() -> None:
    df = fetch_ohlc()
    print(df.head())


if __name__ == "__main__":
    main()
