import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Monte Carlo simulation parameters
np.random.seed(42)  # For reproducibility

# Black-Scholes model parameters
S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)
n_simulations = 10000  # Number of Monte Carlo simulations
n_steps = 252  # Number of time steps (daily steps in a year)

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

def simulate_asset_paths(S0, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths

# Simulate asset price paths
paths = simulate_asset_paths(S0, T, r, sigma, n_simulations, n_steps)

# Compute European Call and Put Option Prices
call_payoff = np.maximum(paths[-1] - K, 0)
put_payoff = np.maximum(K - paths[-1], 0)

# Discount payoffs to present value
call_price_mc = np.exp(-r * T) * np.mean(call_payoff)
put_price_mc = np.exp(-r * T) * np.mean(put_payoff)

# Compute Black-Scholes prices
call_price_bs = BS_CALL(S0, K, T, r, sigma)
put_price_bs = BS_PUT(S0, K, T, r, sigma)

# Print results
print(f"Monte Carlo European Call Option Price: {call_price_mc:.2f}")
print(f"Monte Carlo European Put Option Price: {put_price_mc:.2f}")
print(f"Black-Scholes European Call Option Price: {call_price_bs:.2f}")
print(f"Black-Scholes European Put Option Price: {put_price_bs:.2f}")

# Plot mean and confidence interval of asset price distribution
mean_prices = np.mean(paths, axis=1)
std_prices = np.std(paths, axis=1)
upper_bound = mean_prices + 1.96 * std_prices
lower_bound = mean_prices - 1.96 * std_prices

plt.figure(figsize=(10,5))
plt.plot(mean_prices, label='Mean Asset Price', color='b')
plt.fill_between(range(n_steps + 1), lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
plt.title('Mean Asset Price with Confidence Interval')
plt.xlabel('Time Steps')
plt.ylabel('Asset Price')
plt.legend()
plt.show()

# Plot convergence of option price estimate
plt.figure(figsize=(10,5))
plt.plot(np.cumsum(call_payoff) / np.arange(1, n_simulations + 1), label='Call Price Convergence', color='b')
plt.axhline(call_price_mc, color='r', linestyle='dashed', label='Final Call Price')
plt.title('Monte Carlo Call Option Price Convergence')
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price')
plt.xscale("log")  # Added log scale for x-axis
plt.legend()
plt.show()
