import numpy as np
import copy
from MallClientArrivalSimulator import MallClientArrivalSimulator

def evaluate_alpha(alpha, n_runs, lambda_rates, periods, slots, seed0=50000):
    profits = []
    for run_id in range(n_runs):
        seed_history = seed0 + 2*run_id
        seed_day     = seed0 + 2*run_id + 1

        # Generate same history
        np.random.seed(seed_history)
        sim_hist = MallClientArrivalSimulator(lambda_rates, periods, slots)
        sim_hist.createPreviousDaysData()
        history = copy.deepcopy(sim_hist.demandHistory)

        # Simulate same day with dynamic(alpha)
        np.random.seed(seed_day)
        sim = MallClientArrivalSimulator(lambda_rates, periods, slots)
        sim.demandHistory = history
        m = sim.simulate(dynamic_pricing=True, alpha=alpha, price_floor=0.0, price_cap=None)
        profits.append(m["profit"] if isinstance(m, dict) else m)

    return float(np.mean(profits)), float(np.std(profits, ddof=1))

def grid_search_alpha(alpha_grid, n_runs, lambda_rates, periods, slots):
    best = None
    for a in alpha_grid:
        mean_p, std_p = evaluate_alpha(a, n_runs, lambda_rates, periods, slots)
        print(f"alpha={a:8.3f}  mean_profit={mean_p:10.2f}  std={std_p:10.2f}")
        if best is None or mean_p > best["mean_profit"]:
            best = {"alpha": a, "mean_profit": mean_p, "std_profit": std_p}
    return best

if __name__ == "__main__":
    lambda_rates = [5, 15, 20]
    periods = [8, 8, 8]
    slots = 100

    alpha_grid = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    best = grid_search_alpha(alpha_grid, n_runs=100, lambda_rates=lambda_rates, periods=periods, slots=slots)
    print("\nBest alpha (coarse):", best)

    best_a = best["alpha"]
    refined = np.linspace(best_a*0.6, best_a*1.4, 21)
    best2 = grid_search_alpha(refined, n_runs=200, lambda_rates=lambda_rates, periods=periods, slots=slots)
    print("\nBest alpha (refined):", best2)
