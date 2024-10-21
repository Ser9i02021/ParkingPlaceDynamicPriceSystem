import numpy as np
import matplotlib.pyplot as plt

class LogNormalReservationPricesDistribution():
    def __init__(self, mu, sigma, sample_size) -> None:
        # Initialize the distribution for reservation prices using a Log-Normal distribution.
        # mu: Mean of the log-normal distribution (log of the mean of the underlying normal distribution).
        # sigma: Standard deviation of the underlying normal distribution.
        # sample_size: Number of reservation prices to generate.

        self.mu = mu if mu is not None else 1.5  # Default mean (mu) for the log-normal distribution is 1.5 if not provided.
        self.sigma = sigma if sigma is not None else 0.5  # Default standard deviation (sigma) is 0.5 if not provided.
        self.sample_size = sample_size if sample_size is not None else 1000  # Default sample size is 1000 if not provided.

        # Generate reservation prices based on the log-normal distribution.
        self.reservation_prices = np.random.lognormal(self.mu, self.sigma, self.sample_size)
        # The log-normal distribution is used here to simulate the variety of prices clients are willing to pay.

        # print(self.reservation_prices)  # Uncomment to print the generated reservation prices for debugging.
        
"""
# Optional Plotting (currently commented out)
# This block generates a histogram of the reservation prices to visualize the distribution.
plt.figure(figsize=(10, 6))  # Set the figure size.
plt.hist(reservation_prices, bins=30, edgecolor='black', density=True)  # Plot the histogram.
plt.title('Histogram of Reservation Prices for Parking Spot')  # Set the title of the plot.
plt.xlabel('Reservation Price')  # Label the x-axis.
plt.ylabel('Frequency')  # Label the y-axis.
plt.grid(True)  # Add a grid for better readability.
plt.show()  # Display the plot.
"""

class LogNormalStayingTimesDistribution():
    def __init__(self, mu, sigma, sample_size) -> None:
        # Initialize the distribution for staying times using a Log-Normal distribution.
        # mu: Mean of the log-normal distribution (log of the mean of the underlying normal distribution).
        # sigma: Standard deviation of the underlying normal distribution.
        # sample_size: Number of staying times to generate.

        self.mu = mu if mu is not None else 1.5  # Default mean (mu) for the log-normal distribution is 1.5 if not provided.
        self.sigma = sigma if sigma is not None else 0.5  # Default standard deviation (sigma) is 0.5 if not provided.
        self.sample_size = sample_size if sample_size is not None else 1000  # Default sample size is 1000 if not provided.

        # Generate staying times based on the log-normal distribution.
        self.stayingTimes = np.random.lognormal(self.mu, self.sigma, self.sample_size)

        # Round the staying times to integers, representing whole hours.
        for i in range(len(self.stayingTimes)):
            self.stayingTimes[i] = round(self.stayingTimes[i])

        # print(self.stayingTimes)  # Uncomment to print the generated staying times for debugging.

# Example usage:
LogNormalReservationPricesDistribution(mu=None, sigma=None, sample_size=10)  # Generate 10 sample reservation prices.
LogNormalStayingTimesDistribution(mu=None, sigma=None, sample_size=10)  # Generate 10 sample staying times.
