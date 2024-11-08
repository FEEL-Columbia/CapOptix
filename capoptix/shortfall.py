# shortfall_analysis.py
import pandas as pd
from capoptix.generation.capacity_provider import CapacityProvider
from capoptix.consumption.consumption import Consumption
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
import numpy as np

class ShortfallAnalyzer:
    def __init__(
            self, 
            capacity_provider : CapacityProvider,
            consumption: Consumption,
            transformation : str = None,
        ):
        '''
            Example of how data looks like:-
        
            timestamps = pd.date_range("2024-11-01", periods=5, freq="H")
            data = pd.DataFrame({
                "solar_generation": [10, 12, 14, 13, 15],
                "wind_generation": [5, 7, 6, 8, 5],
                "consumption": [20, 25, 22, 21, 20]
            }, index=timestamps)
        '''
        self.capacity_provider = capacity_provider
        self.consumption = consumption
        self.shortfall_data = self.calculate_shortfall()
        if transformation == "normal":
            self.shortfall_data = self.apply_normal_transformation()
        elif transformation == "lognormal":
            self.shortfall_data = self.apply_lognormal_transformation()

    def calculate_total_generation(self) -> pd.Series:
        """Calculate the total generation using the capacity provider."""
        return self.capacity_provider.get_total_generation()
        
    def calculate_shortfall(self) -> pd.Series:
        """Calculate the shortfall as consumption minus total generation."""
        
        total_generation = self.calculate_total_generation()
        total_demand = self.consumption.get_consumption_data()

        # Calculate shortfall (consumption - total generation)
        shortfall = total_demand - total_generation

        return shortfall
    
    def apply_normal_transformation(self) -> pd.Series:
        quantile_transformer = QuantileTransformer(output_distribution='normal',random_state = 42)
        transformed_data = quantile_transformer.fit_transform(self.shortfall_data.values.reshape(-1,1)).flatten()
        transformed_series = pd.Series(transformed_data, index=self.shortfall_data.index, name="QQ_Shortfall")
        # skewness = transformed_series.skew()
        # print(f"Skewness after QQ Transformation: {skewness:.5f}")

        return transformed_series
    
    def apply_lognormal_transformation(self) -> pd.Series:
        pr = self.shortfall_data
        rank_df = stats.rankdata(pr, method='average')
        n = len(pr)
        ecdf_df = (rank_df - 0.5) / n

        mean_original = pr.mean()
        var_original = pr.var(ddof=0)
        s2 = np.log(var_original / mean_original**2 + 1)
        s = np.sqrt(s2)
        m = np.log(mean_original) - s2 / 2

        transformed_data = stats.lognorm.ppf(ecdf_df, s=s, scale=np.exp(m))
        transformed_series = pd.Series(transformed_data, index=self.shortfall_data.index, name="LogNorm_Shortfall")

        return transformed_series
