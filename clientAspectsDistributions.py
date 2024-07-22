import numpy as np
import matplotlib.pyplot as plt

class LogNormalReservationPricesDistribution():
    def __init__(self, mu, sigma, sample_size) -> None:
        # Parameters for the Log-Normal distribution
        self.mu = mu if mu is not None else 1.5
        self.sigma = sigma if sigma is not None else 0.5
        self.sample_size =  sample_size if sample_size is not None else 1000

        # Generate reservation prices
        self.reservation_prices = np.random.lognormal(self.mu, self.sigma, self.sample_size)
        #print(self.reservation_prices)
"""
# Plot the histogram of the generated reservation prices
plt.figure(figsize=(10, 6))
plt.hist(reservation_prices, bins=30, edgecolor='black', density=True)
plt.title('Histogram of Reservation Prices for Parking Spot')
plt.xlabel('Reservation Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
"""

LogNormalReservationPricesDistribution(mu=None, sigma=None, sample_size=10)
