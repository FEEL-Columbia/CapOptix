import pandas as pd
from .consumption import Consumption
import numpy as np
from capoptix.utils import _infer_sample_time
from capoptix.generation.capacity_provider import CapacityProvider

class ScaledConsumption(Consumption):
    def __init__(self, data: pd.DataFrame, 
                 capacity_provider: CapacityProvider,
                 method: str = 'total'):
        super().__init__(data)
        self.method = method
        self.capacity_provider = capacity_provider
        sample_time = _infer_sample_time(data, timestamp_column="timestamps")
        if sample_time == '15min':
            self.samples_per_day = 96
        elif sample_time == '5min':
            self.samples_per_day = 288
        elif sample_time == 'hourly':
            self.samples_per_day = 24
        else:
            raise ValueError("sample_time must be one of '15min', '5min', or 'hourly'")
    
    def get_consumption_data(self) -> pd.Series:
        """Return the scaled consumption data based on the selected method."""
        if self.method == 'total':
            return self._scale_total()
        elif self.method == 'per_day':
            return self._scale_in_chunks(self.samples_per_day)  
        elif self.method == 'per_week':
            return self._scale_in_chunks(self.samples_per_day * 7) 
        elif self.method == 'per_month':
            return self._scale_in_chunks(self.samples_per_day * 30) 
        elif self.method == 'per_year':
            return self._scale_in_chunks(self.samples_per_day * 365) 
        else:
            raise ValueError("Unsupported scaling method. Use one of total, per_day, per_week, per_month, per_year")

    def _scale_total(self) -> pd.Series:
        """Scale the consumption data using a single scaling factor for the entire dataset."""
        total_generation = self.capacity_provider.get_total_generation().sum()
        total_demand = super().get_consumption_data().sum()
        scaling_factor = total_generation / total_demand
        return super().get_consumption_data() * scaling_factor

    def _scale_in_chunks(self, chunk_size: int) -> pd.Series:
        """Scale the consumption data in chunks (e.g., per day, week, month)."""
        total_generation_series = self.capacity_provider.get_total_generation()
        scaled_values = []
        for i in range(0, len(self.data), chunk_size):
            chunk_gen = total_generation_series[i:i + chunk_size]
            chunk_dem = super().get_consumption_data()[i:i + chunk_size]

            chunk_gen_sum = chunk_gen.sum()
            chunk_dem_sum = chunk_dem.sum()

            scaling_factor = chunk_gen_sum / chunk_dem_sum if chunk_dem_sum != 0 else 0

            if scaling_factor == 0:
                scaled_chunk = chunk_dem  # Avoid division by zero
            else:
                scaled_chunk = chunk_dem * scaling_factor

            scaled_values.extend(scaled_chunk)

        return pd.Series(scaled_values, index=self.data.index[:len(scaled_values)])
