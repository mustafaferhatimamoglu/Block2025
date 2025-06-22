import datetime


def simulate_trades(
    initial_balance: float = 100.0,
    start_date: datetime.date = datetime.date(2025, 3, 1),
    commission: float = 0.002,
):
    """Simulate monthly trades with a commission on each buy and sell."""

    # Hypothetical monthly returns from March to October 2025
    months = [
        ("March", 0.05),
        ("April", -0.03),
        ("May", 0.04),
        ("June", 0.02),
        ("July", -0.01),
        ("August", 0.03),
        ("September", -0.02),
        ("October", 0.04),
    ]

    balance = initial_balance
    history = []

    for month, change in months:
        # Deduct buy commission
        balance -= balance * commission
        # Apply monthly return
        balance *= (1 + change)
        # Deduct sell commission
        balance -= balance * commission
        history.append((month, balance))

    end_date = datetime.date(2025, 10, 31)
    return history, end_date, balance


def main():
    history, end_date, final_balance = simulate_trades()
    for month, bal in history:
        print(f"End of {month}: ${bal:.2f}")
    print(f"\nFinal balance on {end_date:%Y-%m-%d}: ${final_balance:.2f}")


if __name__ == "__main__":
    main()
