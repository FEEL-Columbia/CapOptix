import pandas as pd
from capoptix.shortfall import ShortfallAnalyzer

def test_generation_sources_sum():
    # Mock data with two generation sources (e.g., solar and wind) and consumption
    timestamps = pd.date_range("2024-11-01", periods=5, freq="H")
    data = pd.DataFrame({
        "solar_generation": [10, 12, 14, 13, 15],
        "wind_generation": [5, 7, 6, 8, 5],
        "consumption": [20, 25, 22, 21, 20]
    }, index=timestamps)

    # Initialize ShortfallAnalyzer with the mock data
    analyzer = ShortfallAnalyzer(data)

    # Manually sum generation sources for expected output
    expected_total_generation = data["solar_generation"] + data["wind_generation"]

    # Load generation data and calculate total generation within the analyzer
    analyzer.load_generation_data()
    total_generation = sum(analyzer.generation_sources)
    
    # Test that the total generation is calculated correctly
    pd.testing.assert_series_equal(total_generation, expected_total_generation, check_dtype=False)          