from MallClientArrivalSimulator import MallClientArrivalSimulator

class Main():
    def __init__(self, lambda_rates: list, periods: list, slots: int) -> None:
        # Initialize the Main class with lambda_rates (arrival rates), periods (hours), and slots (total parking slots)
        
        self.lambda_rates = lambda_rates  # Average clients per hour for morning, afternoon, and evening.
        # Example: [5, 15, 20] means lower arrival rate in the morning (5), moderate in the afternoon (15), and higher in the evening (20).
        
        self.periods = periods  # Duration of each period (morning, afternoon, evening). 
        # Example: [8, 8, 8] indicates each period lasts for 8 hours, resulting in a total of 24 hours.

        self.slots = slots  # Total number of parking slots available in the parking lot.

        # Create an instance of the MallClientArrivalSimulator with the initialized parameters.
        simulator = MallClientArrivalSimulator(self.lambda_rates, self.periods, self.slots)

        # This method seems to generate historical data for previous days, which might be useful for initializing the system or simulating future demand.
        simulator.createPreviousDaysData()

        # Run the actual simulation to generate client arrivals based on the lambda rates, periods, and available slots.
        simulator.simulate()

        # Display the simulated hourly client arrivals for inspection. This shows how many clients arrived during each time slice (hour).
        print(f"Hourly arrivals: {simulator.arrivals_per_time_slice}")
        
        # Display the total number of clients that arrived during the entire simulated day.
        print(f"Total arrivals for the day: {simulator.get_total_arrivals()}")

        # Optional: Uncomment the line below to visualize the client arrivals over time in a plot. 
        # This can help better understand how the arrivals are distributed throughout the day.
        # simulator.plot_arrivals()


# Instantiate the Main class with example parameters:
# Lambda rates: [5, 15, 20] -> average number of client arrivals per hour for morning, afternoon, and evening.
# Periods: [8, 8, 8] -> 8-hour periods for morning, afternoon, and evening.
# Slots: 100 -> total number of parking slots available in the parking area.
Main([5, 15, 20], [8, 8, 8], 100)
