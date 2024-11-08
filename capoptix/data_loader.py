import pandas as pd
from .utils import validate_dataframe

class DataLoader:
    '''
    This class would help us in 
    1. Data Loading
    2. Checking if nodal/zonal price data is available in the dataset.
    '''
    def __init__(self, filepath:str):
        self.filepath = filepath
        self.data = None

    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.filepath)
        try:
            validate_dataframe(self.data, required_columns =["timestamps"])
            self.data["timestamps"] = pd.to_datetime(self.data["timestamps"])
            self.data['hour'] = self.data['timestamps'].dt.hour
            self.data['day'] = self.data['timestamps'].dt.day
            self.data['month'] = self.data['timestamps'].dt.month
            self.data['year'] = self.data['timestamps'].dt.year
            self.data['minutes'] = self.data['timestamps'].dt.minute
        except ValueError as e:
            raise ValueError("Error in processing the dataframe: ensure you have a timeseries data with timestamps as one of the column names") from e
        return self.data

    def check_price_data(self) -> bool:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        try:
            validate_dataframe(self.data,required_columns=["nodal_prices"])
            return True
        except ValueError:
            return False    