from abc import ABC, abstractmethod
import pandas as pd

class CapacityProvider(ABC):
    def __init__(
            self,
            data : pd.DataFrame,
            expiration : float = None,
            contract_period : float = None,
            strike_price : float = None, 
    ):
        self.data = data
        self.expiration = expiration
        self.contract_period = contract_period
        self.strike_price = strike_price
        self.sources = []
    
    def add_source(self, source):
        """Add a generation source to the provider."""
        self.sources.append(source)

    def get_total_generation(self) -> pd.Series:
        """Calculate the total generation by summing up all sources."""
        if not self.sources:
            raise ValueError("No generation sources have been added.")
        
        total_generation = pd.Series(0, index=self.data.index)
        for source in self.sources:
            total_generation += source.get_generation_data()

        return total_generation

    def get_parameters(self):
        """Return provider-specific parameters."""
        return {
            "expiration": self.expiration,
            "contract_period": self.contract_period,
            "strike_price": self.strike_price
        }
    
