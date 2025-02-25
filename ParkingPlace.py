import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
    def setPrice(self, demandHistory: list, timeSliceToPredictDemand: int):
        # Dynamically set the slot price based on historical demand and the current number of free parking slots.
        '''
        numFreePSlots = 0  # Counter for free parking slots.
        for ps in self.pSlots:
            if not ps:  # Check if the slot is free.
                numFreePSlots += 1
        '''
        numFreePSlots = len(self.pSlots) - sum(self.pSlots) # Number of free slots
        

        clientArrivalsOnThatHourOnPreviousDays = []  # List to collect arrivals for the specific hour across previous days.
        
        # Loop through the demand history and gather client arrivals for the time slice we want to predict.
        for dayTimeSlice in demandHistory:
            clientArrivalsOnThatHourOnPreviousDays.append(dayTimeSlice[timeSliceToPredictDemand])
        
        # Prepare the data for the linear regression model.
        X = np.array(range(len(clientArrivalsOnThatHourOnPreviousDays))).reshape(-1, 1)  # Use time steps as features.
        y = np.array(clientArrivalsOnThatHourOnPreviousDays).reshape(-1, 1)  # Use past arrivals as target values.

        model = LinearRegression()  # Initialize linear regression model.
        model.fit(X, y)  # Train the model with historical data.

        # Predict the demand for the next time slice, which is the time after the most recent day.
        new_X = np.array([[len(demandHistory)]])  # Predict the demand for the next day.
        prediction = model.predict(new_X)  # Get the predicted demand.

        expectedDemand = round(prediction[0][0])  # Round the predicted demand to an integer value.
        
        # Calculate the price based on the expected demand and the number of free slots.
        # If there are no free slots, avoid division by zero by assigning an infinite price or handling it differently.
        self.slotPrice = expectedDemand / numFreePSlots if numFreePSlots > 0 else float('inf')

    def getPrice(self):
        # Return the current price of a parking slot.
        return self.slotPrice

    def updateFullStatus(self):
        # Update the full status based on whether all parking slots are occupied.
        self.full = all(self.pSlots)  # Set to True if all parking slots are occupied (all True values in pSlots).
