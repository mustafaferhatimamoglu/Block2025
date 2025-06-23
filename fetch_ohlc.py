import argparse
import requests
import pandas as pd
import time


PAIR = "BLOCK_USDT"
INTERVAL = "1h"
DEFAULT_DAYS = 365  # gate.io allows up to 10000 candles (~416 days)


def fetch_ohlc(days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Download hourly OHLCV data for the configured pair from gate.io."""

    def fetch_chunk(start: int) -> list:
        url = "https://api.gateio.ws/api/v4/spot/candlesticks"
        params = {
            "currency_pair": PAIR,
            "interval": INTERVAL,
            "limit": 1000,
            "from": start,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    now = int(time.time())
    start = now - days * 86400
    all_rows: list[list[str]] = []
    while start < now:
        chunk = fetch_chunk(start)
        if not chunk:
            break
        all_rows.extend(chunk)
        last_ts = int(chunk[-1][0])
        start = last_ts + 3600
        time.sleep(0.2)

    columns = [
        "timestamp",
        "quote_volume",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "complete",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df[["open", "high", "low", "close", "volume", "quote_volume"]] = df[
        ["open", "high", "low", "close", "volume", "quote_volume"]
    ].astype(float)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "complete",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch hourly OHLCV data from gate.io and save to CSV"
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
