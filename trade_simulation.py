"""Wrapper to run the trading simulation from :mod:`block_price_prediction`."""

from block_price_prediction import TradeSimulator


def main() -> None:
    simulator = TradeSimulator()
    history, end_date, final_balance = simulator.simulate()
    for month, bal in history:
        print(f"End of {month}: ${bal:.2f}")
    print(f"\nFinal balance on {end_date:%Y-%m-%d}: ${final_balance:.2f}")


if __name__ == "__main__":
    main()
