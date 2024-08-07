import numpy as np
import matplotlib.pyplot as plt
from main import ParkingPlace, Client
from clientAspectsDistributions import LogNormalReservationPricesDistribution, LogNormalStayingTimesDistribution

class MallClientArrivalSimulator:
    def __init__(self, lambda_rates, periods, slots):
        """
        Initializes the simulator with given lambda rates, periods, and number of slots.
        
        :param lambda_rates: List of average arrival rates (lambda) for each period.
        :param periods: List of durations (in hours) for each period.
        :param slots: Total number of parking slots available.
        """
        self.lambda_rates = lambda_rates
        self.periods = periods
        self.total_hours = sum(periods)
        self.slots = slots
        self.arrivals_per_time_slice = None
        self.total_arrivals = 0
        self.prices = None
        self.demandHistory = []
        self.simTime = 0

    def createPreviousDaysData(self):
        for day in range(5):
            """
            Simulates client arrivals for each hour based on the Poisson distribution.
            """
            self.arrivals_per_time_slice = []
            
            # Arrivals per period ((morning, afeternoon, evening) 8 time slices per period)
            for lambda_rate, period in zip(self.lambda_rates, self.periods): 
                arrivalsFromPeriod = np.random.poisson(lambda_rate, period)
                #print()
                #print(arrivalsFromPeriod)

                self.arrivals_per_time_slice.extend(arrivalsFromPeriod)
                #print(self.arrivals_per_time_slice)

            #self.total_arrivals = sum(self.arrivals_per_time_slice)
            
            self.demandHistory.append(self.arrivals_per_time_slice)
        #print(self.demandHistory)
        
        
    def simulate(self):
        # Creates the parking place
        pp = ParkingPlace()
        """
        Simulates client arrivals for each hour based on the Poisson distribution.
        """
        self.arrivals_per_time_slice = []
        
        MorningAfternoonEvening = 0
        # Arrivals per period ((morning, afeternoon, evening) 8 time slices per period)
        for lambda_rate, period in zip(self.lambda_rates, self.periods): 
            arrivalsFromPeriod = np.random.poisson(lambda_rate, period)
            #print(arrivals)

            

            rpDistribution = LogNormalReservationPricesDistribution(mu=None, sigma=None, sample_size=sum(arrivalsFromPeriod))
            stDistribution = LogNormalStayingTimesDistribution(mu=None, sigma=None, sample_size=sum(arrivalsFromPeriod))

            self.arrivals_per_time_slice.extend(arrivalsFromPeriod)
            
            # Iteration covering each slice of time for the current period
            for i in range(1, len(arrivalsFromPeriod) + 1):
                print("instant %d" % self.simTime)
                print()

                # Verfiy if there are clients leaving their respective slots
                ci = 0
                while ci < len(pp.clientsInPP):
                    if pp.clientsInPP[ci].tryFreeParkingSlot(self.simTime):
                        print("client %d freed slot %d" % (pp.clientsInPP[ci].id, pp.clientsInPP[ci].pSlotOccupiedIndex))
                        pp.clientsInPP.remove(pp.clientsInPP[ci])
                        ci -= 1
                    ci += 1



                

                # For each time slice, the parking place will measure the price for a slot based on the demand value in the same time slice on previous days,
                # predicting the demand value for the present time slice
                pp.setPrice(self.demandHistory, MorningAfternoonEvening * len(arrivalsFromPeriod) + (i - 1))                
                
                initial = 0 if i == 1 else arrivalsFromPeriod[i - 2]
                end = arrivalsFromPeriod[i - 1] if i == 1 else arrivalsFromPeriod[i - 1] + arrivalsFromPeriod[i - 2]
                # Each arrival represents a potential client           
                for clientNum in range(initial, end):
                    # Generate the time that each client stays in his/her slot
                    # ...
                    clientTimeStay = stDistribution.stayingTimes[clientNum]
                    
                    # Generate the reservation price for each client
                    clientReservationPrice = rpDistribution.reservation_prices[clientNum]
                    
                    # Creates a new potential client
                    newPotentialClient = Client(self.total_arrivals + clientNum, clientReservationPrice, clientTimeStay, pp)
                    
                    # If a slot is occupied by the new client, the slot price needs to be updated
                    if (newPotentialClient.tryOccupyParkingSlot(pp.getPrice(), self.simTime)):
                        pp.setPrice(self.demandHistory, MorningAfternoonEvening * len(arrivalsFromPeriod) + (i - 1))                
                        
                        # Also add the client to the vector of all active clients in the PP
                        pp.clientsInPP.append(newPotentialClient)

                        print("client %d occupied slot %d" % (newPotentialClient.id, newPotentialClient.pSlotOccupiedIndex))


                    #print("slots' state:")
                    #print(pp.pSlots)
                    print()
                
                # Update simTime every time slice
                self.simTime += 1 

            MorningAfternoonEvening += 1
            self.total_arrivals += sum(arrivalsFromPeriod)
        
        #self.total_arrivals = sum(self.arrivals_per_time_slice)
    
    def plot_arrivals(self):
        """
        Plots the simulated client arrivals per hour.
        """
        if self.arrivals_per_hour is None:
            print("No simulation data available. Run simulate() first.")
            return
        
        plt.bar(range(self.total_hours), self.arrivals_per_hour)
        plt.xlabel('Hour')
        plt.ylabel('Number of Arrivals')
        plt.title('Client Arrivals Per Hour')
        plt.show()
    
    def get_total_arrivals(self):
        """
        Returns the total number of arrivals.
        
        :return: Total number of arrivals.
        """
        if self.total_arrivals is None:
            print("No simulation data available. Run simulate() first.")
            return None
        
        return self.total_arrivals


# Example usage
lambda_rates = [5, 15, 20]  # Average clients per hour for morning, afternoon, evening
periods = [8, 8, 8]  # 8 hours each for morning, afternoon, evening
slots = 100  # Total number of parking slots available

simulator = MallClientArrivalSimulator(lambda_rates, periods, slots)
simulator.createPreviousDaysData()
simulator.simulate()
'''
print(f"Hourly arrivals: {simulator.arrivals_per_hour}")
print(f"Total arrivals for the day: {simulator.get_total_arrivals()}")
print(f"Prices for each parking slot: {simulator.get_prices()}")
#simulator.plot_arrivals()
'''
