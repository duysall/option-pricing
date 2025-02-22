# Option-Pricing

## Overview
This repository contains Python implementations of various option pricing models commonly used in quantitative finance. The goal is to provide efficient and well-documented code for pricing derivatives using Monte Carlo simulations, the Black-Scholes model, and other techniques.

## Implemented Models
### Monte Carlo Simulation for Option Pricing
- Simulates multiple asset price paths using Geometric Brownian Motion (GBM).
- Estimates European call and put option prices based on simulated payoffs.
- Includes convergence visualization to assess accuracy.

### Black-Scholes Model
- Computes theoretical prices for European call and put options.
- Uses closed-form solutions derived from the Black-Scholes formula.
- Compares results with Monte Carlo simulations.

## Installation & Dependencies
Ensure Python 3.7+ is installed. Clone the repo and install the required dependencies:
```bash
git clone https://github.com/yourusername/Option-Pricing.git
cd Option-Pricing
pip install -r requirements.txt
