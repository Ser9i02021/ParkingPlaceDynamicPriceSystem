from ParkingPlace import ParkingPlace

class Client:
    def __init__(self, id, reservationPrice: int, timeStay: int, pp: ParkingPlace):
        # Initialize the client with a unique ID, reservation price (max amount the client is willing to pay),
        # timeStay (duration of parking), and a reference to the parking place (pp) the client is using.

        self.id = id  # Unique identifier for the client.
        self.reservationPrice = reservationPrice  # Maximum price the client is willing to pay for a parking slot.
        self.valuePaid = 0  # Amount the client will pay, initialized to 0.
        self.pSlotOccupiedIndex = -1  # Index of the occupied parking slot, initialized to -1 meaning no slot is occupied.
        self.timeStay = timeStay  # Duration for which the client will stay parked.
        self.timeEntrance = 0  # Time when the client enters the parking lot, initialized to 0.
        # self.timeExit = 0  # (Optional) You could track the exit time if necessary.
        self.pp = pp  # Reference to the ParkingPlace object that the client is using.

    def tryOccupyParkingSlot(self, slotPrice: int, presentTime: int):
        # Attempt to occupy a parking slot if the price is within the client's reservation price.

        if not self.pp.full:  # Only attempt if the parking lot is not full.
            #print("slot price: %f" % slotPrice)  # Debug: Print the current slot price.
            #print("client %d reservation price: %f" % (self.id, self.reservationPrice))  # Debug: Print the client's reservation price.
            
            if slotPrice <= self.reservationPrice:  # Check if the slot price is affordable for the client.
                # Loop through the parking slots to find a free one.
                for i, slot in enumerate(self.pp.pSlots):
                    if not slot:  # If the slot is available (False), occupy it.
                        self.pp.pSlots[i] = True  # Mark the slot as occupied (True).
                        self.pSlotOccupiedIndex = i  # Store the index of the occupied slot.
                        self.pp.updateFullStatus()  # Update the parking lot's full status (in case it's now full).
                        self.timeEntrance = presentTime  # Record the time the client entered the parking lot.
                        self.valuePaid = slotPrice  # The client pays the current slot price.
                        return True  # Occupation was successful.
        return False  # Return False if occupation fails (lot is full or price too high).

    def tryFreeParkingSlot(self, timePresent: int):
        # Attempt to free the parking slot once the client's timeStay is complete.

        if self.pSlotOccupiedIndex > -1:  # Ensure the client actually occupies a slot.
            # Check if the client's parking duration has been met.
            if self.timeStay == timePresent - self.timeEntrance:
                self.valuePaid = self.valuePaid * self.timeStay  # Calculate the total cost based on time stayed.
                self.pp.pSlots[self.pSlotOccupiedIndex] = False  # Free the occupied slot (set to False).
                # self.pSlotOccupiedIndex = -1  # (Needs to be commented to print the correct index of the freed slot) You could reset this value if needed after freeing the slot.
                self.pp.profit += self.valuePaid  # Add the client's payment to the parking lot's profit.
                self.pp.updateFullStatus()  # Update the parking lot's full status (in case it's no longer full).
                return True  # Freeing the slot was successful.
        return False  # Return False if the slot is not freed.
