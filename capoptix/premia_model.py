import numpy as np
from scipy.stats import norm
import scipy as sp

class CapacityPremiaModel:
    def __init__(self, model_type='bachelier'):
        """
        Initialize the capacity premia model with a specified type.
        
        Parameters:
        - model_type (str): The type of model to use ('bachelier', 'blackscholes', 'jump_diffusion_blackscholes', 'jump_diffusion_bachelier').
        """
        self.model_type = model_type

    def _bachelier(self, sigma, S, K, r, t):
        d = (S * np.exp(r * t) - K) / np.sqrt((sigma**2 / (2 * r)) * (np.exp(2 * r * t) - 1))
        C = np.exp(-r * t) * (S * np.exp(r * t) - K) * norm.cdf(d) + np.exp(-r * t) * np.sqrt((sigma**2 / (2 * r)) * (np.exp(2 * r * t) - 1)) * norm.pdf(d)
        return C

    def _black_scholes(self, sigma, S, K, r, t):
        d1 = np.multiply((np.log(S / K) + (r + 0.5 * sigma ** 2) * t), 1. / (sigma * np.sqrt(t)))
        d2 = d1 - sigma * np.sqrt(t)
        C = np.multiply(S,norm.cdf(d1)) - np.multiply(K * np.exp(-r * t), norm.cdf(d2))
        return C

    def _jump_diffusion_black_scholes(self, S, K, T, r, sigma, mu_J, sigma_J, lambda_jump, max_jumps=50):
        price = 0
        k = np.exp(mu_J + (sigma_J**2) / 2) - 1

        for n in range(max_jumps):
            sigma_n = np.sqrt(sigma**2 + ((n * sigma_J**2) / T))
            bs_price = self._black_scholes(sigma_n, S, K, r, T)
            poisson_prob = np.exp(-lambda_jump * T) * (lambda_jump * T) ** n / sp.special.factorial(n)
            price += poisson_prob * bs_price

        return price

    def monte_carlo_total_jump_diffusion_bachelier(self, S0, K, T, tau, r, sigma, lambda_j, mu_j, sigma_j, N=100000):
        np.random.seed(42)
        option_prices = []
        maturity_values = []
        premium = 0
        hours_in_a_year = 365 * 24

        for time in range(tau):
            maturity = (time / hours_in_a_year) + T
            time_steps = 1
            dt = maturity/time_steps
            paths = np.zeros((N, time_steps + 1))
            paths[:, 0] = S0

            dW = np.random.normal(0, np.sqrt(dt), N)
            dN = np.random.poisson(lambda_j*dt,N)
            total_jump_contributions = np.array(
                [(np.sum(np.random.normal(mu_j, sigma_j, n))/(n)) if n > 0 else 0 for n in dN]
            )

            paths[:,time_steps] = (
                S0 + 
                r * dt + 
                sigma * dW + 
                total_jump_contributions
            )

            payoffs = np.maximum(paths[:,-1] - K, 0)
            option_price = np.exp(-r * T) * np.mean(payoffs)
            premium += option_price
            option_prices.append(option_price)
            maturity_values.append(premium)

        return premium, option_prices, maturity_values

    def premia_calculation(self, S, K, r, t, tau, **kwargs):
        """
        Premia using the specified model type.
        
        Parameters:
        - S (float): Initial stock price.
        - K (float): Strike price.
        - r (float): Risk-free rate.
        - t (float): Current time in years.
        - tau (int): Maturity period in hours or steps.
        - kwargs: Additional parameters for specific models (e.g., sigma, lambda_jump, mu_J, sigma_J, etc.).
        
        Returns:
        - float, list, list: The premium, list of option prices, and maturity values.
        """
        if self.model_type == 'bachelier':
            return self.bachelier_total(kwargs['sigma'], S, K, r, t, tau)
        elif self.model_type == 'blackscholes':
            return self.blackscholes_final(kwargs['sigma'], S, K, r, t, tau)
        elif self.model_type == 'jump_diffusion_blackscholes':
            return self.total_jump_diffusion_call(S, K, t, tau, r, kwargs['sigma'], kwargs['mu_J'], kwargs['sigma_J'], kwargs['lambda_jump'], kwargs.get('max_jumps', 50))
        elif self.model_type == 'jump_diffusion_bachelier':
            return self.monte_carlo_total_jump_diffusion_bachelier(S, K, t, tau, r, kwargs['sigma'], kwargs['lambda_j'], kwargs['mu_j'], kwargs['sigma_j'])
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    # Wrapper methods for total calculations (similar to existing functions you provided)
    def bachelier_total(self, sigma, energy_price, K, r, t, tau):
        premium, option_prices, maturity_values = 0, [], []
        hours_in_a_year = 365 * 24
        for time in range(tau):
            maturity = (time / hours_in_a_year) + t
            option_price = self._bachelier(sigma, energy_price, K, r, maturity)
            premium += option_price
            option_prices.append(option_price)
            maturity_values.append(premium)
        return premium, option_prices, maturity_values

    def blackscholes_final(self, sigma, energy_price, K, r, t, tau):
        premium, option_prices, maturity_values = 0, [], []
        hours_in_a_year = 365 * 24
        for time in range(tau):
            maturity = (time / hours_in_a_year) + t
            option_price = self._black_scholes(sigma, energy_price, K, r, maturity)
            premium += option_price
            option_prices.append(option_price)
            maturity_values.append(premium)
        return premium, option_prices, maturity_values

    def total_jump_diffusion_call(self, energy_price, K, t, tau, r, sigma, mu_J, sigma_J, lambda_jump, max_jumps):
        premium, option_prices, maturity_values = 0, [], []
        hours_in_a_year = 365 * 24
        for time in range(tau):
            maturity = (time / hours_in_a_year) + t
            option_price = self._jump_diffusion_black_scholes(energy_price, K, maturity, r, sigma, mu_J, sigma_J, lambda_jump, max_jumps)
            premium += option_price
            option_prices.append(option_price)
            maturity_values.append(premium)
        return premium, option_prices, maturity_values
