import numpy as np
from MallClientArrivalSimulator import MallClientArrivalSimulator

def grid_search_best_fixed_price(
    lambda_rates: list,
    periods: list,
    slots: int,
    price_grid: list[float],
    n_runs_per_price: int = 200,
    seed0: int = 12345,
    n_history_days: int = 5,
):
    """
    Returns:
      best_price, best_mean_profit, full_results
    full_results is a list of dicts: {"price": ..., "mean_profit": ..., "std_profit": ...}
    """

    results = []

    for p in price_grid:
        profits = []

        # Common random numbers: each run_id uses the same seed across ALL prices
        # => fairer comparisons + less Monte-Carlo noise
        for run_id in range(n_runs_per_price):
            np.random.seed(seed0 + run_id)

            sim = MallClientArrivalSimulator(lambda_rates, periods, slots)
            sim.createPreviousDaysData(n_days=n_history_days)

            profit = sim.simulate(dynamic_pricing=False, fixed_price=float(p))
            profits.append(profit)

        mean_profit = float(np.mean(profits))
        std_profit = float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0

        results.append({"price": float(p), "mean_profit": mean_profit, "std_profit": std_profit})
        print(f"price={p:8.3f}  mean_profit={mean_profit:12.2f}  std={std_profit:10.2f}")

    best = max(results, key=lambda d: d["mean_profit"])
    return best["price"], best["mean_profit"], results


if __name__ == "__main__":
    lambda_rates = [5, 15, 20]
    periods = [8, 8, 8]
    slots = 100

    # --- Coarse grid (example) ---
    # If your reservation price distribution has mean around ~5, this spans below/above that.
    price_grid = np.linspace(0.5, 12.0, 24)

    best_p, best_profit, results = grid_search_best_fixed_price(
        lambda_rates, periods, slots,
        price_grid=price_grid,
        n_runs_per_price=1000,   # use 200 for search; use 1000 later for final comparison
        seed0=42,
        n_history_days=5
    )

    print("\nBest fixed price (coarse):", best_p, "mean profit:", best_profit)

    # --- Optional refine step around the best ---
    fine_grid = np.linspace(max(0.01, best_p - 1.0), best_p + 1.0, 41)

    best_p2, best_profit2, results2 = grid_search_best_fixed_price(
        lambda_rates, periods, slots,
        price_grid=fine_grid,
        n_runs_per_price=300,
        seed0=42,
        n_history_days=5
    )

    print("\nBest fixed price (refined):", best_p2, "mean profit:", best_profit2)
