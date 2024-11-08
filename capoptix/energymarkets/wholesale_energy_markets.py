# wholesale_energy_markets.py
import pandas as pd
from croniter import croniter
from datetime import datetime, timedelta

class WholesaleEnergyMarket:
    def __init__(self, data: pd.DataFrame, market_name: str, bidding_cron: str, scale:bool = False):
        """
        Base class for wholesale energy markets.
        
        Parameters:
        - data (pd.DataFrame): The data relevant to the market (e.g., market prices, demand, etc.).
        - market_name (str): The name of the market (e.g., 'DAM', 'RTM').
        - bidding_cron (str): Cron string representing the bidding schedule.
        """
        self.data = data
        self.market_name = market_name
        self.bidding_cron = bidding_cron
        self.scale = scale

    def scaling_prices(self, data) -> pd.Series:
        # Calculate the 25th and 75th percentiles
        q1 = data.quantile(0.001)
        q3 = data.quantile(0.999)

        below_q1_count = (data < q1).sum()
        above_q3_count = (data > q3).sum()
        total_out_of_range_count = below_q1_count + above_q3_count
        # print(total_out_of_range_count)

        # Replace values below Q1 with Q1 and above Q3 with Q3
        data_modified = data.apply(lambda x: 0 if x < 0 else q3 if x > q3 else x)
        return data_modified


    def get_market_data(self) -> pd.Series:
        """
        Retrieve the market data associated with this market.

        Returns:
        - pd.DataFrame: The DataFrame containing the market data.
        """
        if "prices" not in self.data.columns:
            raise ValueError("Price data not found.")
        price_series = self.data["prices"]
        if self.scale:
            return self.scaling_prices(price_series)
        else:
            return price_series

    def next_bidding_time(self, start_time: datetime = None) -> datetime:
        """
        Calculate the next bidding time based on the cron string.

        Parameters:
        - start_time (datetime, optional): The time from which to start calculating the next bidding time.
                                           Defaults to current time if not provided.

        Returns:
        - datetime: The next bidding time.
        """
        if start_time is None:
            start_time = datetime.now()

        cron_schedule = croniter(self.bidding_cron, start_time)
        return cron_schedule.get_next(datetime)

    def is_bidding_time(self, current_time: datetime = None) -> bool:
        """
        Check if the current time matches a bidding time based on the cron string.

        Parameters:
        - current_time (datetime, optional): The current time to check. Defaults to now if not provided.

        Returns:
        - bool: True if it matches the bidding schedule, False otherwise.
        """
        if current_time is None:
            current_time = datetime.now()

        next_bid_time = self.next_bidding_time(current_time - timedelta(minutes=1))
        return current_time == next_bid_time


class DAM(WholesaleEnergyMarket):
    def __init__(self, data: pd.DataFrame,scale:bool = False):
        """
        Day-Ahead Market (DAM) class.
        
        Parameters:
        - data (pd.DataFrame): The data relevant to the DAM.
        """
        super().__init__(data, market_name="DAM", bidding_cron="0 12 * * *", scale= scale)  # Example: Bids at 12:00 daily


class RTM(WholesaleEnergyMarket):
    def __init__(self, data: pd.DataFrame,scale:bool = False):
        """
        Real-Time Market (RTM) class.
        
        Parameters:
        - data (pd.DataFrame): The data relevant to the RTM.
        """
        super().__init__(data, market_name="RTM", bidding_cron="*/30 * * * *", scale=scale)  # Example: Bids every 30 minutes
