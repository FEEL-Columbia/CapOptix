# generation/solar.py
import pandas as pd

class SolarGeneration:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_generation_data(self) -> pd.Series:
        """Extract and return solar generation data."""
        if 'solar_generation' not in self.data.columns:
            raise ValueError("Solar generation data not found.")
        return self.data['solar_generation']