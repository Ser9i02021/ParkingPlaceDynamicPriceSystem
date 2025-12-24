import numpy as np
import matplotlib.pyplot as plt
from ParkingPlace import ParkingPlace
from Client import Client
from ClientAspectsDistributions import LogNormalReservationPricesDistribution, LogNormalStayingTimesDistribution

class MallClientArrivalSimulator:
    def __init__(self, lambda_rates, periods, slots):
        """
        Initializes the simulator with given lambda rates, periods, and number of slots.
        
        :param lambda_rates: List of average arrival rates (lambda) for each period (morning, afternoon, evening).
        :param periods: List of durations (in hours) for each period (e.g., [8, 8, 8]).
        :param slots: Total number of parking slots available.
        """
        self.lambda_rates = lambda_rates  # Average client arrival rates (Poisson distributed).
        self.periods = periods  # Duration of each period (in hours).
        self.total_hours = sum(periods)  # Total number of hours for the entire day (e.g., 24).
        self.slots = slots  # Total number of parking slots available.
        self.arrivals_per_time_slice = None  # List to store the arrivals per hour.
        self.total_arrivals = 0  # Track the total number of client arrivals for the day.
        self.demandHistory = []  # Store historical demand data for dynamic pricing prediction.
        self.simTime = 0  # Simulation time, representing the current time slice.

    def createPreviousDaysData(self, n_days: int = 5, mu: float = 1.5, sigma: float = 0.5, history_price: float = 3.45):
        """
        Builds demand history from historical *occupation outcomes* (served + rejected_full),
        then converts to a price-normalized arrival intensity lambda_hat per time slice.
        """

        self.demandHistory = []  # now this will store lambda_hat vectors per day

        for day in range(n_days):
            # Important: reset RNG state outside if you want reproducibility (in your evaluation you already do)
            sim = MallClientArrivalSimulator(self.lambda_rates, self.periods, self.slots)

            # We simulate a "historical day" under a fixed pricing regime so history is not circular
            # May be worth using dynamic pricing, but let's keep it simple for now
            metrics = sim.simulate(dynamic_pricing=False, fixed_price=history_price)

            served = metrics["served_per_slice"]              # list length = total_hours
            rej_full = metrics["rejected_full_per_slice"]     # list length = total_hours
            price = metrics["posted_price_per_slice"]         # constant == history_price, but stored anyway

            lambda_hat_day = []
            pp_tmp = ParkingPlace(self.slots)
            for t in range(len(served)):
                willing = served[t] + rej_full[t]
                S = pp_tmp._lognormal_survival(price[t], mu, sigma)

                # Avoid divide-by-zero if S is extremely tiny
                if S < 1e-9:
                    lam = 0.0
                else:
                    lam = willing / S

                lambda_hat_day.append(lam)

            self.demandHistory.append(lambda_hat_day)

    def simulate(self, dynamic_pricing=True, fixed_price=3.45, alpha=1.0, price_floor=0.0, price_cap=None, tau0=0.8, kappa=0.1):
        # -------------------------
        # NEW: Per-time-slice series (needed for robust history)
        # -------------------------
        served_per_slice = []              # how many clients actually parked in each time slice
        rejected_full_per_slice = []       # how many would have parked but lot was full (per slice)
        posted_price_per_slice = []        # the "posted" price at the start of each time slice

        
        """
        Simulates client arrivals and slot occupations for a single day, using Poisson-distributed arrivals.

        Now also returns metrics:
        - served_clients
        - rejected_due_to_price
        - rejected_due_to_full
        - avg_occupancy_rate (slot-hours used / total slot-hours)
        - avg_price (mean posted price per time slice)
        - price_volatility (std dev of posted price per time slice)
        """

        # (IMPORTANT if you run many simulations) reset simulation time for a fresh "day"
        self.simTime = 0

        # Initialize the parking place with the specified number of slots.
        pp = ParkingPlace(self.slots)

        self.arrivals_per_time_slice = []  # List to store client arrivals per time slice.

        MorningAfternoonEvening = 0  # Period counter (to track morning, afternoon, evening periods).

        # -------------------------
        # NEW: Metrics Counters
        # -------------------------
        served_clients = 0
        rejected_due_to_price = 0
        rejected_due_to_full = 0

        # Occupancy accounting (slot-hours used / total slot-hours)
        slot_hours_used = 0
        total_slot_hours = self.slots * sum(self.periods)

        # Price series for average price and volatility
        # (We record 1 "posted" price per time slice, i.e., the price after the update at the start of the hour.)
        posted_prices = []

        # Generation of arrivals for each period (Morning, Afternoon and Evening)
        for lambda_rate, period in zip(self.lambda_rates, self.periods):  # Iteration over Morning, Afternoon and Evening
            arrivalsFromPeriod = np.random.poisson(lambda_rate, period)  # Generate arrivals based on Poisson distribution.
                                                                        # E.g.: [4, 6, 7, 3, 2, 4, 8, 5]
            self.arrivals_per_time_slice.extend(arrivalsFromPeriod)  # Append the arrivals from the current period to the list.

        self.total_arrivals = sum(self.arrivals_per_time_slice)  # Get the total arrivals for the day

        # Distribute client reservation prices and staying times using log-normal distributions for all clients on the day
        rpDistribution = LogNormalReservationPricesDistribution(mu=None, sigma=None, sample_size=sum(self.arrivals_per_time_slice))
        stDistribution = LogNormalStayingTimesDistribution(mu=None, sigma=None, sample_size=sum(self.arrivals_per_time_slice))

        acc = 0  # Accumulation variable for clients's ID generation and correct staying times and reservation prices attributions

        for period in range(len(self.periods)):  # Iteration over Morning, Afternoon and Evening

            # Loop through each time slice (hour) in the current period (morning/afternoon/evening).
            # Index "i" refers to the current hour of the current period
            for i in range(1, self.periods[period] + 1):
                # NEW: reset per-slice counters
                served_this_slice = 0
                rejected_full_this_slice = 0

                # Check if any clients are ready to leave (free their parking slots).
                ci = 0
                while ci < len(pp.clientsInPP):
                    if pp.clientsInPP[ci].tryFreeParkingSlot(self.simTime):
                        pp.clientsInPP.remove(pp.clientsInPP[ci])  # Remove the client who left the parking lot.
                        ci -= 1  # Adjust index after removal.
                    ci += 1

                # Compute the global time slice index for this hour in the day:
                # (morning offset + hour index inside period)
                time_slice_idx = sum(self.periods[:MorningAfternoonEvening]) + (i - 1)

                # For each time slice, set/define the price:
                # - dynamic pricing: predict the demand for that slice (hour) based on historical data
                # - fixed pricing: keep the same price for the whole simulation
                if dynamic_pricing:
                    # Avoid division by zero inside setPrice if the parking is full
                    # (setPrice divides by number of free slots)
                    if not pp.full:
                        pp.setPriceScarcityMarkup(self.demandHistory,
                        time_slice_idx,
                        mu=1.5,
                        sigma=0.5,
                        p_star=3.45,
                        tau0=tau0,
                        kappa=kappa,
                        p_min=0.1,
                        p_cap=20.0
                        )
                        '''
                        pp.setPriceRevenueMax(
                        self.demandHistory,
                        time_slice_idx,
                        mu=1.5,
                        sigma=0.5,
                        p_min=0.1,
                        p_cap=20.0
                        )
                        '''
                    # else: keep last price when full (no update possible)

                    effective_price = pp.getPrice()
                else:
                    # In fixed mode, we always use fixed_price
                    effective_price = fixed_price
                    # (Optional but useful) keep pp internal price consistent
                    pp.slotPrice = fixed_price

                # NEW: store the posted price for this time slice (for avg price and volatility metrics)
                posted_prices.append(float(effective_price))

                # NEW: store the posted price for this time slice (for historical reconstruction)
                posted_price_per_slice.append(float(effective_price))


                # Determine the range of clients arriving in this time slice (hour).
                initial = acc
                end = self.arrivals_per_time_slice[time_slice_idx] + acc

                # Handle client arrivals for the current time slice (hour).
                for clientNum in range(initial, end):
                    clientTimeStay = stDistribution.stayingTimes[clientNum]  # Time each client stays.
                    clientReservationPrice = rpDistribution.reservation_prices[clientNum]  # Client reservation price.

                    # (Optional robustness) ensure staying time is at least 1 time slice
                    # clientTimeStay = max(1, int(clientTimeStay))

                    # Create a new potential client.
                    newPotentialClient = Client(clientNum, clientReservationPrice, clientTimeStay, pp)

                    # -------------------------
                    # NEW: classify outcome (served / rejected due to full / rejected due to price)
                    # -------------------------
                    # Note: price may change within the hour when dynamic pricing updates after a successful occupation,
                    # so in dynamic mode we always re-read the current price before evaluating a client.
                    if dynamic_pricing:
                        effective_price = pp.getPrice()
                    else:
                        effective_price = fixed_price

                    if pp.full:
                        rejected_due_to_full += 1
                        rejected_full_this_slice += 1   # NEW
                        continue
                    else:
                        rejected_due_to_full += 1
                        rejected_full_this_slice += 1   # NEW


                    if effective_price > clientReservationPrice:
                        rejected_due_to_price += 1
                        continue

                    # Try to occupy a parking slot with the effective price
                    if newPotentialClient.tryOccupyParkingSlot(effective_price, self.simTime):
                        served_clients += 1
                        served_this_slice += 1          # NEW

                        # If dynamic pricing is enabled, update the parking slot price after occupation
                        # (keeps your original behavior).
                        # Optionally you can comment this out to have price updates only at the start of each hour
                        if dynamic_pricing:
                            if not pp.full:  # guard if parking became full due to this occupation
                                pp.setPriceScarcityMarkup(self.demandHistory,
                                time_slice_idx,
                                mu=1.5,
                                sigma=0.5,
                                p_star=3.45,
                                tau0=tau0,
                                kappa=kappa,
                                p_min=0.1,
                                p_cap=20.0
                                )
                                
                                '''
                                pp.setPriceRevenueMax(
                                self.demandHistory,
                                time_slice_idx,
                                mu=1.5,
                                sigma=0.5,
                                p_min=0.1,
                                p_cap=20.0
                                )
                                '''
                        # Add client to active client list
                        pp.clientsInPP.append(newPotentialClient)
                    else:
                        # Fallback (should be rare): treat as full
                        rejected_due_to_full += 1

                # NEW: occupancy accounting for this time slice (slot-hours)
                # After processing departures + arrivals, current occupancy represents slot usage until next time slice.
                occupied_slots = int(np.sum(pp.pSlots))
                slot_hours_used += occupied_slots

                # NEW: finalize per-slice series for this time slice
                served_per_slice.append(served_this_slice)
                rejected_full_per_slice.append(rejected_full_this_slice)

                self.simTime += 1  # Increment the simulation time after each time slice (hour).
                acc = end

            MorningAfternoonEvening += 1  # Move to the next period (afternoon/evening).

        # -------------------------
        # NEW: Final Metrics
        # -------------------------
        avg_occupancy_rate = (slot_hours_used / total_slot_hours) if total_slot_hours > 0 else 0.0

        # Average posted price and volatility across time slices (one price value per time slice)
        avg_price = float(np.mean(posted_prices)) if len(posted_prices) > 0 else 0.0
        price_volatility = float(np.std(posted_prices)) if len(posted_prices) > 1 else 0.0  # population std dev

        # Return profit + metrics (profit is still included for compatibility with your previous analysis)
        return {
            "profit": pp.profit,
            "served_clients": served_clients,
            "rejected_due_to_price": rejected_due_to_price,
            "rejected_due_to_full": rejected_due_to_full,
            "avg_occupancy_rate": avg_occupancy_rate,
            "avg_price": avg_price,
            "price_volatility": price_volatility,

            # NEW: per-slice series for createPreviousDaysData()
            "served_per_slice": served_per_slice,
            "rejected_full_per_slice": rejected_full_per_slice,
            "posted_price_per_slice": posted_price_per_slice,
        }


    def plot_arrivals(self):
        """
        Plots the simulated client arrivals per hour.
        """
        if self.arrivals_per_time_slice is None:
            print("No simulation data available. Run simulate() first.")
            return
        
        # Create a bar plot to show client arrivals per hour.
        plt.bar(range(self.total_hours), self.arrivals_per_time_slice)
        plt.xlabel('Hour')
        plt.ylabel('Number of Arrivals')
        plt.title('Client Arrivals Per Hour')
        plt.show()
    
    def get_total_arrivals(self):
        """
        Returns the total number of arrivals for the simulation.
        """
        if self.total_arrivals is None:
            print("No simulation data available. Run simulate() first.")
            return None
        
        return self.total_arrivals
