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


def summarize(profits_dyn, profits_fix):
    profits_dyn = np.array(profits_dyn, dtype=float)
    profits_fix = np.array(profits_fix, dtype=float)
    diff = profits_dyn - profits_fix

    n = len(diff)
    mean_dyn = profits_dyn.mean()
    mean_fix = profits_fix.mean()
    mean_diff = diff.mean()

    sd_diff = diff.std(ddof=1) if n > 1 else 0.0
    se_diff = sd_diff / np.sqrt(n) if n > 1 else 0.0

    # Normal approx CI is fine for n ~ 1000
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff

    win_rate = (diff > 0).mean()
    rel_uplift = (mean_diff / mean_fix * 100.0) if mean_fix != 0 else np.nan

    return {
        "n_runs": n,
        "mean_profit_dynamic": mean_dyn,
        "mean_profit_fixed": mean_fix,
        "mean_diff_dyn_minus_fixed": mean_diff,
        "ci95_diff": (ci_low, ci_high),
        "win_rate_dyn_gt_fixed": win_rate,
        "relative_uplift_percent": rel_uplift,
    }


if __name__ == "__main__":
    lambda_rates = [5, 15, 20]
    periods = [8, 8, 8]
    slots = 100

    N = 1000  # final evaluation; you can start with 100 to test quickly

    profits_dyn = []
    profits_fix = []

    for run_id in range(N):
        pdyn, pfix = run_one_pair(
            run_id=run_id,
            lambda_rates=lambda_rates,
            periods=periods,
            slots=slots,
            tau0=0.8,
            kappa=0.1,
            seed0=100000,  # keep fixed for reproducibility; change if you want a new test set
        )
        profits_dyn.append(pdyn)
        profits_fix.append(pfix)

    summary = summarize(profits_dyn, profits_fix)

    print("\n=== Paired Evaluation: Dynamic vs Fixed(3.45) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
