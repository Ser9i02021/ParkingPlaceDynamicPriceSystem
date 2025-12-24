import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import math
from statistics import NormalDist

class ParkingPlace():
    def __init__(self, numPSlots: int, pSlots: list = None, slotPrice: int = 0, full: bool = False, profit: int = 0) -> None:
        # Initialize ParkingPlace with flexible number of parking slots (numPSlots), 
        # an optional list of slots (pSlots), initial slot price, full status, and profit.

        self.numPSlots = numPSlots  # Number of parking slots, making the model adaptable to different lot sizes.
        self.pSlots = pSlots if pSlots is not None else np.zeros(self.numPSlots, bool)  # Initialize parking slots if not provided.
        self.slotPrice = slotPrice  # Initial price for parking slots.
        self.full = full  # Indicates if the parking place is full (boolean).
        self.profit = profit  # Accumulated profit.
        self.clientsInPP = []  # List to track clients currently in the parking lot.

    def reset(self):
        # Reset the parking lot by clearing the parking slots, resetting the full status, and profit.
        self.pSlots = np.zeros(self.numPSlots, bool)  # Resets the parking slots to all empty.
        self.full = False  # Marks the parking place as not full.
        self.profit = 0  # Resets the profit to 0.

    # Time slice refers to 1 of the 24 hours of the day
    def setPrice(
    self,
    demandHistory: list,
    timeSliceToPredictDemand: int,
    alpha: float = 1.0,
    price_floor: float = 0.0,
    price_cap: float | None = None
    ):
        # Dynamically set the slot price based on historical demand, 
        # the current number of free parking slots and Alpha parameter.
        numFreePSlots = len(self.pSlots) - sum(self.pSlots) # Number of free slots

        # If full, pricing is not really meaningful; keep it capped or very high
        if numFreePSlots == 0:
            self.slotPrice = float("inf") if price_cap is None else float(price_cap)
            return

        # Collect arrivals at this time slice across previous days
        clientArrivalsOnThatHourOnPreviousDays = []
        for dayTimeSlice in demandHistory:
            clientArrivalsOnThatHourOnPreviousDays.append(dayTimeSlice[timeSliceToPredictDemand])

        # Fit linear regression: X = day index, y = arrivals at that hour
        X = np.array(range(len(clientArrivalsOnThatHourOnPreviousDays))).reshape(-1, 1)
        y = np.array(clientArrivalsOnThatHourOnPreviousDays).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        # Predict next day (index = len(history))
        new_X = np.array([[len(demandHistory)]])
        prediction = model.predict(new_X)

        expectedDemand = int(round(prediction[0][0]))
        expectedDemand = max(0, expectedDemand)  # demand can't be negative

        # Raw dynamic price with scaling
        raw_price = alpha * (expectedDemand / numFreePSlots)

        # Apply floor / cap
        price = max(price_floor, raw_price)
        if price_cap is not None:
            price = min(price, float(price_cap))

        self.slotPrice = price

    def getPrice(self):
        # Return the current price of a parking slot.
        return self.slotPrice

    def updateFullStatus(self):
        # Update the full status based on whether all parking slots are occupied.
        self.full = all(self.pSlots)  # Set to True if all parking slots are occupied (all True values in pSlots).

    def _lognormal_survival(self, p: float, mu: float, sigma: float) -> float:
        """P(WTP >= p) for LogNormal(mu, sigma)."""
        if p <= 0:
            return 1.0
        z = (math.log(p) - mu) / (sigma * math.sqrt(2))
        return 0.5 * (1.0 - math.erf(z))  # 1 - CDF

    def _compute_monopoly_price(self, mu: float, sigma: float, p_min: float, p_max: float, n_grid: int = 2000) -> float:
        """
        Computes p* that maximizes p * P(WTP >= p) (capacity-slack monopoly price),
        via a simple grid search (fast enough and SciPy-free).
        """
        prices = np.linspace(p_min, p_max, n_grid)
        surv = np.array([self._lognormal_survival(p, mu, sigma) for p in prices])
        obj = prices * surv
        return float(prices[int(np.argmax(obj))])

    def setPriceRevenueMax(
        self,
        demandHistory: list,
        timeSliceToPredictDemand: int,
        mu: float = 1.5,
        sigma: float = 0.5,
        p_min: float = 0.1,
        p_cap: float = 20.0
    ):
        """
        Dynamic pricing: choose the price that maximizes expected revenue this time slice
        given:
        - predicted arrivals (from demandHistory)
        - free slots right now
        - WTP ~ LogNormal(mu, sigma)

        Uses an analytic "capacity-aware" rule:
        - If capacity is slack: price = monopoly price p*
        - If capacity is tight: raise price so that expected willing ≈ freeSlots
        """

        # Count free slots (supply)
        freeSlots = int(np.sum(~self.pSlots))

        # If full, set to cap (or keep last); no sales possible anyway
        if freeSlots <= 0:
            self.slotPrice = float(p_cap)
            return

        # Forecast arrivals for this hour from history
        # (In your simulated world there is no across-day trend, so the MEAN is usually better than linear regression.)
        hist = [day[timeSliceToPredictDemand] for day in demandHistory]
        demand_hat = float(np.mean(hist))
        demand_hat = max(0.0, demand_hat)

        # If no demand expected, go to floor
        if demand_hat <= 0.0:
            self.slotPrice = float(p_min)
            return

        # 1) compute monopoly price p* (cache it so we don't recompute every hour)
        cache_key = (mu, sigma, p_min, p_cap)
        if not hasattr(self, "_pstar_cache") or self._pstar_cache["key"] != cache_key:
            p_star = self._compute_monopoly_price(mu, sigma, p_min, p_cap)
            surv_star = self._lognormal_survival(p_star, mu, sigma)
            self._pstar_cache = {"key": cache_key, "p_star": p_star, "surv_star": surv_star}
        else:
            p_star = self._pstar_cache["p_star"]
            surv_star = self._pstar_cache["surv_star"]

        # 2) capacity check at monopoly price
        expected_willing_at_pstar = demand_hat * surv_star

        if expected_willing_at_pstar <= freeSlots:
            #print("OK")
            # slack capacity -> standard monopoly price
            price = p_star

            #print(price)
        else:
            #print("TIGHT")
            # tight capacity -> raise price until expected_willing ≈ freeSlots
            # We want: demand_hat * survival(price) = freeSlots  => survival(price) = freeSlots / demand_hat
            s_target = freeSlots / demand_hat  # in (0,1)

            # Convert survival target to CDF target: CDF = 1 - survival
            cdf_target = 1.0 - s_target
            cdf_target = min(max(cdf_target, 1e-6), 1.0 - 1e-6)

            z = NormalDist().inv_cdf(cdf_target)  # standard normal quantile
            price = math.exp(mu + sigma * z)      # lognormal quantile

            #print(price)

        # clamp to safety bounds
        price = max(p_min, min(float(price), float(p_cap)))

        #print("actual price: ", price)
        self.slotPrice = price

    def setPriceScarcityMarkup(
    self,
    demandHistory: list,
    timeSlice: int,
    mu: float = 1.5,
    sigma: float = 0.5,
    p_star: float = 3.45,     # you can compute/cache this once like you already did
    tau0: float = 0.8,        # scarcity threshold
    kappa: float = 0.1,       # surge strength
    p_min: float = 0.1,
    p_cap: float = 20.0
):
        freeSlots = int(np.sum(~self.pSlots))
        if freeSlots <= 0:
            self.slotPrice = p_cap
            return

        # demandHistory now stores lambda_hat vectors per day
        lam_hist = [day[timeSlice] for day in demandHistory]
        lam_hat = float(np.mean(lam_hist))
        lam_hat = max(0.0, lam_hat)

        # OPTIONAL (better): use expected willing at base price p_star
        S_star = self._lognormal_survival(p_star, mu, sigma)
        willing_hat = lam_hat * S_star

        # tightness = expected willing / freeSlots
        tau = willing_hat / max(1.0, float(freeSlots))

        markup = 1.0 + kappa * max(0.0, tau - tau0)
        price = p_star * markup

        self.slotPrice = max(p_min, min(float(price), float(p_cap)))
