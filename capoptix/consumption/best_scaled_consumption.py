# consumption/best_scaled_consumption.py
import pandas as pd
from .scaled_consumption import ScaledConsumption
from scipy.stats import gaussian_kde
import numpy as np

class BestScaledConsumption(ScaledConsumption):
    def __init__(self, data: pd.DataFrame, capacity_provider, methods=None):
        super().__init__(data, capacity_provider)
        # List of methods to try for scaling; default is all common methods
        self.methods = methods if methods else ['total', 'per_day', 'per_week', 'per_month', 'per_year']

    def get_consumption_data(self) -> pd.Series:
        """Determine and return the best scaled consumption data based on KDE overlap."""
        best_method = None
        max_overlap = -np.inf
        best_scaled_data = None

        # Generate KDE for the total generation data
        kde_total_gen = gaussian_kde(self.capacity_provider.get_total_generation().dropna())

        # Iterate over each method to find the best one based on KDE overlap
        for method in self.methods:
            self.method = method
            scaled_data = super().get_consumption_data()

            if scaled_data.isnull().any():
                scaled_data = scaled_data.dropna()

            # Generate KDE for the scaled consumption data
            kde_scaled_data = gaussian_kde(scaled_data)

            # Calculate overlap between the total generation KDE and the scaled consumption KDE
            overlap = self._calculate_kde_overlap(kde_total_gen, kde_scaled_data)

            # Update if a better overlap is found
            if overlap > max_overlap:
                max_overlap = overlap
                best_method = method
                best_scaled_data = scaled_data

        print(f"Best scaling method: {best_method} with KDE overlap of {max_overlap:.4f}")
        return best_scaled_data

    def _calculate_kde_overlap(self, kde1, kde2, num_points=1000):
        """Calculate the overlap between two KDEs."""
        x_min = min(kde1.dataset.min(), kde2.dataset.min())
        x_max = max(kde1.dataset.max(), kde2.dataset.max())
        x = np.linspace(x_min, x_max, num_points)

        kde1_vals = kde1(x)
        kde2_vals = kde2(x)

        overlap = np.trapz(np.minimum(kde1_vals, kde2_vals), x)
        return overlap
