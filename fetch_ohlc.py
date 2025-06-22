
import argparse
import requests
import pandas as pd
import time


COIN_ID = "blockasset"
VS_CURRENCY = "usd"
DEFAULT_DAYS = 365  # max history allowed by public API


def fetch_ohlc(days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Download hourly OHLC data for the configured coin."""

    def fetch_prices(start: int, end: int) -> list:
        url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart/range"
        params = {"vs_currency": VS_CURRENCY, "from": start, "to": end}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()["prices"]

    now = int(time.time())
    start = now - days * 86400
    all_prices = []
    while start < now:
        end = min(start + 90 * 86400, now)
        all_prices.extend(fetch_prices(start, end))
        start = end

    df = pd.DataFrame(all_prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("date", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    ohlc = df["price"].resample("1H").ohlc()
    return ohlc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OHLC data for Blockasset and save to CSV"
    )
    parser.add_argument(
        "--outfile",
        "-o",
        default="ohlc.csv",
        help="Output CSV file (default: ohlc.csv)",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of days of history to fetch (max 365)",
    )
    args = parser.parse_args()

    df = fetch_ohlc(days=min(args.days, DEFAULT_DAYS))
    df.to_csv(args.outfile)
    print(df.head())
    print(f"Saved {len(df)} rows to {args.outfile}")


if __name__ == "__main__":
    main()
