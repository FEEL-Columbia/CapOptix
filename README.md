# CapOptix

CapOptix is a Python library designed to model, simulate, and analyze energy markets and capacity premia using advanced mathematical techniques. It provides tools to handle energy consumption and generation data, price nodal energy options, and model wholesale energy markets such as Day-Ahead Market (DAM) and Real-Time Market (RTM). CapOptix is structured to facilitate modular usage, making it easy for users to customize and integrate specific components into their projects.

## Folder Structure

```
CapOptix/
|-- consumption/
|   |-- consumption.py
|   |-- scaled_consumption.py
|   |-- best_scaled_consumption.py
|
|-- generation/
|   |-- capacity_provider.py
|   |-- solar.py
|   |-- wind.py
|
|-- energymarket/
|   |-- wholesaleenergymarket.py
|
|-- dataloader.py
|-- nodal_price_generator.py
|-- premia_model.py
|-- utils.py
```

### Description of Modules

- **consumption/**
  - `consumption.py`: Handles loading and managing consumption data.
  - `scaled_consumption.py`: Provides functionality for scaling consumption data using various methods.
  - `best_scaled_consumption.py`: Selects the best scaling approach based on specific metrics.

- **generation/**
  - `capacity_provider.py`: Represents a class for handling the total generation from multiple sources.
  - `solar.py`: Handles solar generation data.
  - `wind.py`: Handles wind generation data.

- **energymarket/**
  - `wholesaleenergymarket.py`: Base class for modeling wholesale energy markets and also include implementation of Day-Ahead Market (DAM) and Real-Time Market (RTM) functionality.

- **dataloader.py**: A module for loading data from CSVs and other file formats.
- **nodal_price_generator.py**: Generates nodal price predictions and model fits using various statistical methods.
- **premia_model.py**: Contains the `CapacityPremiaModel` class for pricing capacity options using different financial models.
- **utils.py**: Utility functions for data validation, handling missing values, plotting, and more.

## Infrastructure Overview

CapOptix leverages a modular architecture to support various components of energy market analysis, as depicted below:

![Infrastructure Diagram](/images/CapacityPremia.pdf)

## Installation and Setup

To use CapOptix, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CapOptix.git
   cd CapOptix
   ```

2. **Install dependencies**:
   Make sure you have Python 3.8 or higher installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the library**:
   Ensure that all modules are in your Python path. You can run CapOptix as a standalone library or integrate it into your projects.

## Example Usage

The following is an example of how to use CapOptix to analyze energy data and calculate capacity premia:

```python
from capoptix.dataloader import DataLoader
from capoptix.consumption.consumption import Consumption
from capoptix.generation.capacity_provider import CapacityProvider
from capoptix.premia_model import CapacityPremiaModel

# Load data
loader = DataLoader(file_path='path/to/your/Energy_AT.csv')
data = loader.load_data()

# Analyze consumption data
consumption = Consumption(data)
scaled_data = consumption.get_scaled_consumption_data(method='daily')

# Analyze generation data
provider = CapacityProvider(sources=["solar", "wind"])
total_generation = provider.get_total_generation()

# Calculate capacity premia
premia_model = CapacityPremiaModel(model_type='blackscholes')
premium, option_prices, maturity_values = premia_model.premia_calculation(
    S=100, K=110, r=0.05, t=0.1, tau=24*365, sigma=0.2
)
print("Premium:", premium)
```

## Example Notebook
For a full working example, refer to the [notebook](/scenario1_low_shortfall.ipynb) provided in the repository.

## Contribution
We welcome contributions! If you'd like to improve CapOptix, please fork the repository and submit a pull request.

--

## License
This project is licensed under the MIT License. See the LICENSE file for more details.



