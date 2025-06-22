from datetime import datetime

# Starting parameters
starting_balance = 100.0
commission_rate = 0.002  # 0.2% per buy and per sell
start_date = datetime(2025, 3, 1)

# Hypothetical monthly returns (percent change)
monthly_returns = {
    '2025-03': 0.05,
    '2025-04': -0.03,
    '2025-05': 0.04,
    '2025-06': 0.02,
}

balance = starting_balance

for month_str, change in monthly_returns.items():
    # Commission when entering trade
    buy_commission = balance * commission_rate
    invested = balance - buy_commission

    # Apply monthly return
    invested *= (1 + change)

    # Commission when exiting trade
    sell_commission = invested * commission_rate
    balance = invested - sell_commission

    print(f"After {month_str}, balance: ${balance:.2f}")

print(f"Final balance as of {list(monthly_returns.keys())[-1]}: ${balance:.2f}")
