import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

class Consumption:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_consumption_data(self) -> pd.Series:
        """Extract and return consumption data."""
        if 'consumption' not in self.data.columns:
            raise ValueError("Consumption data not found.")
        return self.data['consumption']