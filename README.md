# Dynamic Pricing System for a Mall Parking Place

## Overview

This project aims to implement a dynamic pricing system for a mall parking place, simulating client arrivals, parking slot occupation, and dynamically adjusting parking prices based on real-time demand. The system leverages a combination of machine learning models, statistical distributions, and time-based simulations to optimize parking slot allocation and pricing.

## Key Components

### 1. Dynamic Pricing Model
The core of the system is a dynamic pricing mechanism that adjusts parking slot prices based on predicted demand. The pricing formula considers the number of free parking slots and predicted demand for each time slice (hourly). Prices increase with higher demand and fewer available slots, ensuring a balanced parking lot usage throughout the day.

### 2. Poisson-Distributed Client Arrivals
Client arrivals are simulated using a Poisson distribution, which is suitable for modeling the random arrival of clients at various periods of the day (morning, afternoon, evening). The lambda rates (average clients per hour) for each period drive the simulation and can be customized based on real-world data.

### 3. Log-Normal Distributed Client Attributes
Clients’ reservation prices (the maximum price they are willing to pay for a parking spot) and staying times (duration of parking) are modeled using a log-normal distribution. This distribution helps simulate the natural variability in client behavior, with some clients willing to pay more and stay longer than others.

### 4. Parking Place Management
The `ParkingPlace` class manages parking slots, dynamically updates prices, and tracks clients currently using the parking lot. It monitors whether the parking lot is full and handles profit calculation based on the time clients occupy their slots.

### 5. Client Behavior Simulation
The `Client` class models each client’s behavior, attempting to occupy a parking slot if the price is within their budget and freeing the slot when their time is up. Each client has a unique reservation price and time to stay, determined by the log-normal distributions.

### 6. Historical Demand Data for Prediction
A crucial aspect of the dynamic pricing system is its ability to predict future demand based on historical data. The system simulates previous days of parking usage and stores demand history. A linear regression model predicts the demand for each upcoming time slice, allowing the system to adjust parking prices dynamically.

### 7. Simulation Engine
The `MallClientArrivalSimulator` class orchestrates the entire simulation, handling the flow of time, client arrivals, and parking slot allocation. It also generates historical data for previous days to feed into the dynamic pricing model. The simulation engine ensures clients enter and exit the parking lot as expected while keeping track of total arrivals and plotting the results.

## How the System Works

### Step 1: Historical Data Generation
The simulator creates historical demand data by simulating parking lot usage for several previous days. This data is used to predict future demand.

### Step 2: Real-Time Simulation
Each hour (or time slice), the simulator generates new clients using Poisson-distributed arrivals and assigns them reservation prices and staying times based on log-normal distributions.

### Step 3: Slot Allocation and Pricing
For each time slice, the parking lot adjusts its prices based on the predicted demand and current availability. Clients decide whether to occupy a parking slot based on the current price. As clients leave after their designated stay time, slots are freed, and the system updates the parking status.

### Step 4: Profit Calculation
As clients pay for their parking based on their stay time and the dynamic pricing model, the parking lot tracks the total profit accumulated.

## Features

### 1. Flexible Simulation Setup
The system allows customization of arrival rates, periods, parking slots, and client attributes.

### 2. Dynamic Pricing
Parking slot prices adjust in real time based on demand predictions, optimizing parking usage and revenue.

### 3. Machine Learning Integration
Linear regression is used to predict future demand based on past arrivals, improving pricing decisions.

### 4. Visualization
The system can visualize client arrivals over time using bar plots, providing insights into parking usage patterns.

## Future Improvements

### 1. Advanced Demand Prediction
Implementing more sophisticated machine learning models like time-series forecasting (ARIMA, LSTM) could improve demand predictions.

### 2. More Complex Client Behavior
Adding features such as early exits or extending stays based on real-world behaviors could enhance the simulation's realism.

### 3. Dynamic Time Granularity
Allowing for time slices smaller than one hour (e.g., half-hour) could provide a more granular view of parking usage.

### 4. Optimization
Introducing optimization algorithms for better price setting and client management would further increase parking lot efficiency.
