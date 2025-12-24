import numpy as np
import copy
from MallClientArrivalSimulator import MallClientArrivalSimulator

FIXED_PRICE = 3.45
MU = 1.5
SIGMA = 0.5

def run_one_pair(run_id, lambda_rates, periods, slots, tau0, kappa, seed0=90000):
    """
    Paired run:
      - same history + same day draws
      - compare dynamic(tau0,kappa) vs fixed(FIXED_PRICE)
    Returns: profit_dynamic, profit_fixed
    """
    seed_history = seed0 + 2 * run_id
    seed_day     = seed0 + 2 * run_id + 1

    # 1) Build history once (occupation-based lambda_hat) with fixed history price
    np.random.seed(seed_history)
    sim_hist = MallClientArrivalSimulator(lambda_rates, periods, slots)
    sim_hist.createPreviousDaysData(n_days=5, mu=MU, sigma=SIGMA, history_price=FIXED_PRICE)
    demand_history = copy.deepcopy(sim_hist.demandHistory)

    # 2) Dynamic with scarcity markup (same day seed)
    np.random.seed(seed_day)
    sim_dyn = MallClientArrivalSimulator(lambda_rates, periods, slots)
    sim_dyn.demandHistory = demand_history
    m_dyn = sim_dyn.simulate(dynamic_pricing=True, fixed_price=FIXED_PRICE, tau0=tau0, kappa=kappa)
    profit_dyn = m_dyn["profit"] if isinstance(m_dyn, dict) else m_dyn

    # 3) Fixed baseline (same day seed)
    np.random.seed(seed_day)
    sim_fix = MallClientArrivalSimulator(lambda_rates, periods, slots)
    sim_fix.demandHistory = demand_history
    m_fix = sim_fix.simulate(dynamic_pricing=False, fixed_price=FIXED_PRICE)
    profit_fix = m_fix["profit"] if isinstance(m_fix, dict) else m_fix

    return profit_dyn, profit_fix


def eval_combo(tau0, kappa, n_runs, lambda_rates, periods, slots, seed0=90000):
    profits_dyn = []
    profits_fix = []

    for run_id in range(n_runs):
        pd, pf = run_one_pair(run_id, lambda_rates, periods, slots, tau0, kappa, seed0=seed0)
        profits_dyn.append(pd)
        profits_fix.append(pf)

    profits_dyn = np.array(profits_dyn, dtype=float)
    profits_fix = np.array(profits_fix, dtype=float)
    diff = profits_dyn - profits_fix

    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd_diff / np.sqrt(n) if n > 1 else 0.0
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    return {
        "tau0": float(tau0),
        "kappa": float(kappa),
        "mean_profit_dynamic": float(profits_dyn.mean()),
        "mean_profit_fixed": float(profits_fix.mean()),
        "mean_diff": mean_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "win_rate": float((diff > 0).mean()),
    }


def grid_search(tau0_grid, kappa_grid, n_runs, lambda_rates, periods, slots):
    results = []
    best = None

    for tau0 in tau0_grid:
        for kappa in kappa_grid:
            r = eval_combo(tau0, kappa, n_runs, lambda_rates, periods, slots)
            results.append(r)

            print(
                f"tau0={tau0:4.2f} kappa={kappa:4.2f}  "
                f"mean_diff={r['mean_diff']:8.3f}  "
                f"CI=[{r['ci95_low']:8.3f}, {r['ci95_high']:8.3f}]  "
                f"win={r['win_rate']:.3f}"
            )

            if best is None or r["mean_diff"] > best["mean_diff"]:
                best = r

    return best, results


if __name__ == "__main__":
    lambda_rates = [5, 15, 20]
    periods = [8, 8, 8]
    slots = 100

    # Coarse grid (fast)
    tau0_grid = [0.60, 0.70, 0.80, 0.90, 1.00]
    kappa_grid = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    best, results = grid_search(tau0_grid, kappa_grid, n_runs=200,  # 200 is enough for tuning
                                lambda_rates=lambda_rates, periods=periods, slots=slots)

    print("\nBest (coarse):", best)

    # Refine around best
    tau0_best = best["tau0"]
    kappa_best = best["kappa"]

    tau0_refined = np.clip(np.linspace(tau0_best - 0.15, tau0_best + 0.15, 7), 0.30, 1.50)
    kappa_refined = np.clip(np.linspace(kappa_best * 0.6, kappa_best * 1.4, 9), 0.01, 5.0)

    best2, results2 = grid_search(tau0_refined, kappa_refined, n_runs=400,
                                  lambda_rates=lambda_rates, periods=periods, slots=slots)

    print("\nBest (refined):", best2)
