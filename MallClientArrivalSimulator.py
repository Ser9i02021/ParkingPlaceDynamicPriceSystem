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

    def createPreviousDaysData(self):
        """
        Simulates previous days' demand history (5 days of client arrivals) to be used for demand prediction.
        """
        for day in range(5):
            self.arrivals_per_time_slice = []  # Reset hourly arrivals for each day.
            
            # Generate arrivals per time period (morning, afternoon, evening) using the Poisson distribution.
            for lambda_rate, period in zip(self.lambda_rates, self.periods): 
                arrivalsFromPeriod = np.random.poisson(lambda_rate, period)
                self.arrivals_per_time_slice.extend(arrivalsFromPeriod)  # Append period arrivals to the time slice list.
            
            self.demandHistory.append(self.arrivals_per_time_slice)  # Store the arrivals for demand prediction.

    def simulate(self):
        """
        Simulates client arrivals and slot occupations for a single day, using Poisson-distributed arrivals.
        """
        # Initialize the parking place with the specified number of slots.
        pp = ParkingPlace(self.slots)

        self.arrivals_per_time_slice = []  # List to store client arrivals per time slice.

        MorningAfternoonEvening = 0  # Period counter (to track morning, afternoon, evening periods).

        # Loop through periods (morning, afternoon, evening) and generate arrivals.
        for lambda_rate, period in zip(self.lambda_rates, self.periods): 
            arrivalsFromPeriod = np.random.poisson(lambda_rate, period)  # Generate arrivals based on Poisson distribution.
            
            # Distribute client reservation prices and staying times using log-normal distributions.
            rpDistribution = LogNormalReservationPricesDistribution(mu=None, sigma=None, sample_size=sum(arrivalsFromPeriod))
            stDistribution = LogNormalStayingTimesDistribution(mu=None, sigma=None, sample_size=sum(arrivalsFromPeriod))

            self.arrivals_per_time_slice.extend(arrivalsFromPeriod)  # Append the arrivals to the list.
            
            # Loop through each time slice in the current period (morning/afternoon/evening).
            for i in range(1, len(arrivalsFromPeriod) + 1):
                print(f"instant {self.simTime}")

                # Check if any clients are ready to leave (free their parking slots).
                ci = 0
                while ci < len(pp.clientsInPP):
                    if pp.clientsInPP[ci].tryFreeParkingSlot(self.simTime):
                        print(f"client {pp.clientsInPP[ci].id} freed slot {pp.clientsInPP[ci].pSlotOccupiedIndex}")
                        pp.clientsInPP.remove(pp.clientsInPP[ci])  # Remove the client who left the parking lot.
                        ci -= 1  # Adjust index after removal.
                    ci += 1

                # For each time slice, predict the demand for that slice based on historical data.
                pp.setPrice(self.demandHistory, MorningAfternoonEvening * len(arrivalsFromPeriod) + (i - 1))

                # Determine the range of clients arriving in this time slice.
                initial = 0 if i == 1 else arrivalsFromPeriod[i - 2]
                end = arrivalsFromPeriod[i - 1] if i == 1 else arrivalsFromPeriod[i - 1] + arrivalsFromPeriod[i - 2]

                # Handle client arrivals for the current time slice.
                for clientNum in range(initial, end):
                    clientTimeStay = stDistribution.stayingTimes[clientNum]  # Time each client stays.
                    clientReservationPrice = rpDistribution.reservation_prices[clientNum]  # Client reservation price.
                    
                    # Create a new potential client.
                    newPotentialClient = Client(self.total_arrivals + clientNum, clientReservationPrice, clientTimeStay, pp)
                    
                    # Try to occupy a parking slot. If successful, update the parking slot price and add the client to active list.
                    if newPotentialClient.tryOccupyParkingSlot(pp.getPrice(), self.simTime):
                        pp.setPrice(self.demandHistory, MorningAfternoonEvening * len(arrivalsFromPeriod) + (i - 1))  # Update price after occupation.
                        pp.clientsInPP.append(newPotentialClient)  # Add client to active client list.
                        print(f"client {newPotentialClient.id} occupied slot {newPotentialClient.pSlotOccupiedIndex}")

                self.simTime += 1  # Increment the simulation time after each time slice.

            MorningAfternoonEvening += 1  # Move to the next period (afternoon/evening).
            self.total_arrivals += sum(arrivalsFromPeriod)  # Update the total arrivals count.
        
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
