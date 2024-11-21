# generation/fossil.py
import pandas as pd

class FossilFuelGeneration:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_generation_data(self) -> pd.Series:
        """Extract and return fossil generation data."""
        if 'fossil_generation' not in self.data.columns:
            raise ValueError("fossil generation data not found.")
        return self.data['fossil_generation']