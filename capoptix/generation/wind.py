# generation/wind.py
import pandas as pd

class WindGeneration:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_generation_data(self) -> pd.Series:
        """Extract and return wind generation data."""
        if 'wind_generation' not in self.data.columns:
            raise ValueError("Wind generation data not found.")
        return self.data['wind_generation']