import random
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class ParkingPlace():
    def __init__(self, pSlots: list = None, slotPrice: int = 0, full: bool = False, profit: int = 0) -> None:
        self.pSlots = pSlots if pSlots is not None else np.zeros(100, bool)
        self.slotPrice = slotPrice
        self.full = full
        self.profit = profit
        self.clientsInPP = []

    def reset(self):
        self.pSlots = np.zeros(10, bool)
        self.full = False
        self.profit = 0

    def setPrice(self, demandHistory: list, timeSliceToPredictDemand: int):
        numFreePSlots = 0
        for ps in self.pSlots:
            if not ps:
                numFreePSlots += 1

        clientArrivalsOnThatHourOnPreviousDays = []
        #print(demandHistory)
        #print(hourToPredictDemand)
        for dayTimeSlice in demandHistory:
            clientArrivalsOnThatHourOnPreviousDays.append(dayTimeSlice[timeSliceToPredictDemand])
            
        
        #print(clientArrivalsOnThatHourOnPreviousDays)
        X = np.array(range(len(clientArrivalsOnThatHourOnPreviousDays))).reshape(-1, 1)
        y = np.array(clientArrivalsOnThatHourOnPreviousDays).reshape(-1, 1)

        model = LinearRegression()
        #print(X)
        #print(y)
        model.fit(X, y)

        new_X = np.array([[5]])
        prediction = model.predict(new_X)

        #print("Predicted next element:", prediction[0][0])

        expectedDemand = round(prediction[0][0])
        #print(expectedDemand)
        self.slotPrice = expectedDemand / numFreePSlots

    def getPrice(self):
        return self.slotPrice

    def updateFullStatus(self):
        self.full = all(self.pSlots)


class Client:
    def __init__(self, id, reservationPrice: int, timeStay: int, pp: ParkingPlace):
        self.id = id
        self.reservationPrice = reservationPrice
        self.valuePaid = 0
        self.pSlotOccupiedIndex = -1
        self.timeStay = timeStay
        self.timeEntrance = 0
        #self.timeExit = 0
        self.pp = pp

    def tryOccupyParkingSlot(self, slotPrice: int, presentTime: int):
        if not self.pp.full:
            print("slot price: %f" % slotPrice)
            print("client reservation price: %f" % self.reservationPrice)
            if slotPrice <= self.reservationPrice:
                for i, slot in enumerate(self.pp.pSlots):
                    if not slot:
                        self.pp.pSlots[i] = True
                        self.pSlotOccupiedIndex = i
                        self.pp.updateFullStatus()
                        self.timeEntrance = presentTime
                        self.valuePaid = slotPrice
                        return True
        return False

    def tryFreeParkingSlot(self, timePresent: int):
        if self.pSlotOccupiedIndex > -1:
            if self.timeStay == timePresent - self.timeEntrance:
                self.valuePaid = self.valuePaid * self.timeStay
                self.pp.pSlots[self.pSlotOccupiedIndex] = False
                #self.pSlotOccupiedIndex = -1
                self.pp.profit += self.valuePaid
                self.pp.updateFullStatus()
                return True
        return False

