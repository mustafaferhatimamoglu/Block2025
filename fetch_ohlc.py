import argparse
import requests
import pandas as pd


ALLOWED_DAYS = {1, 7, 14, 30, 90, 180, 365, "max"}


def fetch_ohlc(coin_id: str, vs_currency: str = "usd", days: str | int = 30) -> pd.DataFrame:
    if isinstance(days, str) and days.isdigit():
        days = int(days)
    if days not in ALLOWED_DAYS:
        allowed = ", ".join(str(d) for d in ALLOWED_DAYS)
        raise ValueError(f"days must be one of {allowed}")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLC data from CoinGecko")
    parser.add_argument("coin_id", help="Coin id on CoinGecko, e.g. bitcoin")
    parser.add_argument(
        "--days",
        default=30,
        help="Number of days (1/7/14/30/90/180/365/max)",
    )
    parser.add_argument("--vs", default="usd", help="Quote currency (default: usd)")
    parser.add_argument("--outfile", help="Optional CSV file to save results")
    args = parser.parse_args()

    df = fetch_ohlc(args.coin_id, vs_currency=args.vs, days=args.days)
    if args.outfile:
        df.to_csv(args.outfile)
        print(f"Saved {len(df)} rows to {args.outfile}")
    else:
        print(df.head())


if __name__ == "__main__":
    main()
