# nodal_price_generator.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

class NodalPriceGenerator:
    def __init__(self, X, Y):
        """
        Initialize the NodalPriceGenerator class with data and a model choice.
        
        Parameters:
        - X (np.array or pd.Series): The supply shortfall data.
        - Y (np.array or pd.Series): The energy price data.
        """
        self.X = X
        self.Y = Y
        self.models = {}
        self.model_choice = None
        self.fitted_predictions = {}

    def evaluation(self, true, pred):
        """Evaluate the model's performance using R^2 score."""
        r2score = r2_score(true, pred)
        print(f"R^2 score: {r2score:.4f}")
        return r2score

    def fit_exponential(self):
        """Fit data to an exponential model."""
        def exponential_model(S_s, alpha, beta):
            return alpha * np.exp(beta * S_s)
        
        params, _ = curve_fit(exponential_model, self.X, self.Y, p0=[1, 0.01])
        alpha, beta = params
        self.models['exponential'] = params
        predicted_prices = exponential_model(self.X, alpha, beta)
        self.fitted_predictions['exponential'] = predicted_prices
        
        # print(f"Exponential model parameters: alpha = {alpha:.4f}, beta = {beta:.4f}")
        return predicted_prices

    def fit_logistic(self):
        """Fit data to a logistic model."""
        def logistic_model(S_s, L, beta, S_0):
            return L / (1 + np.exp(-beta * (S_s - S_0)))
        
        params, _ = curve_fit(logistic_model, self.X, self.Y, p0=[1, 0.01, 10])
        L, beta, S_0 = params
        self.models['logistic'] = params
        predicted_prices = logistic_model(self.X, L, beta, S_0)
        self.fitted_predictions['logistic'] = predicted_prices
        
        # print(f"Logistic model parameters: L = {L:.4f}, beta = {beta:.4f}, S_0 = {S_0:.4f}")
        return predicted_prices

    def fit_tanh(self):
        """Fit data to a tanh model."""
        def tanh_model(S_s, L, beta, S_0):
            return L * np.tanh(beta * (S_s - S_0))
        
        params, _ = curve_fit(tanh_model, self.X, self.Y, p0=[1, 0.01, 10])
        L, beta, S_0 = params
        self.models['tanh'] = params
        predicted_prices = tanh_model(self.X, L, beta, S_0)
        self.fitted_predictions['tanh'] = predicted_prices
        
        # print(f"Tanh model parameters: L = {L:.4f}, beta = {beta:.4f}, S_0 = {S_0:.4f}")
        return predicted_prices

    def fit_linear(self):
        """Fit data to a linear model using OLS."""
        X_ = sm.add_constant(self.X)
        model = sm.OLS(self.Y, X_).fit()
        predicted_prices = model.predict(X_)
        self.models['linear'] = model.params
        self.fitted_predictions['linear'] = predicted_prices
        
        # print("Linear model parameters:", model.params)
        return predicted_prices

    def fit_quantile_regression(self):
        """Fit data using quantile regression for various quantiles."""
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        X_ = sm.add_constant(self.X)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X, self.Y, label="Data", alpha=0.5)

        predicted_prices_list = []
        for quantile in quantiles:
            model = sm.QuantReg(self.Y, X_)
            res = model.fit(q=quantile)
            predicted_prices = res.predict(X_)
            predicted_prices_list.append(predicted_prices)
            plt.plot(self.X, predicted_prices, label=f'Quantile {quantile}', lw=2)

        plt.title('Quantile Regression for Different Quantiles')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

        best_r2 = -np.inf
        best_index = -1
        for idx, predicted_prices in enumerate(predicted_prices_list):
            r2 = self.evaluation(np.asarray(self.Y), np.asarray(predicted_prices))
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_index = idx

        best_predicted_prices = predicted_prices_list[best_index]
        self.models['quantile'] = f'Best quantile {quantiles[best_index]}'
        self.fitted_predictions['quantile'] = best_predicted_prices
        
        # print(f"Best quantile: {quantiles[best_index]} with R^2 = {best_r2:.4f}")
        return best_predicted_prices

    def fit_model(self, model_choice):
        self.model_choice =model_choice
        """Automatically fit the model specified by model_choice."""
        if model_choice == 'exponential':
            return self.fit_exponential()
        elif model_choice == 'logistic':
            return self.fit_logistic()
        elif model_choice == 'tanh':
            return self.fit_tanh()
        elif model_choice == 'linear':
            return self.fit_linear()
        elif model_choice == 'quantile':
            return self.fit_quantile_regression()
        else:
            raise ValueError(f"Invalid model choice: {model_choice}")

    def plot_model_fit(self):
        """Plot the observed data and the chosen model fit."""
        if self.model_choice not in self.fitted_predictions:
            print(f"Model '{self.model_choice}' not fitted yet.")
            return

        plt.figure(figsize=(8, 5))
        plt.scatter(self.X, self.Y, label='Observed Prices', edgecolors="k")
        plt.plot(self.X, self.fitted_predictions[self.model_choice], color='red', label=f'Fitted {self.model_choice} Model')
        plt.xlabel('Supply Shortfall (MW)', fontsize=14, fontweight="bold")
        plt.ylabel('Energy Price (EUR/MWh)', fontsize=14, fontweight="bold")
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(prop={'weight': 'bold', 'size': 14})
        plt.title(f'{self.model_choice} Model Fit')
        plt.show()
