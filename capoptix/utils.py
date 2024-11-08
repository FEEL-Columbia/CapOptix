import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm,probplot
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import font_manager as fm


def validate_dataframe(data:pd.DataFrame, required_columns:list):
    '''
    Validate that the Dataframe contains the required columns.
    '''
    missing_columns = []
    for col in required_columns:
        if col not in data.columns:
            missing_columns.append(col)

    if missing_columns:
        raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}")

def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series to have values between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min())

def log_message(message: str, level: str = "INFO"):
    """Log a message with a specified level (INFO, WARNING, ERROR)."""
    print(f"[{level}] {message}")

def _infer_sample_time(data: pd.DataFrame, timestamp_column: str = None) -> str:
    """
    Infer the sample time of a dataset by analyzing the time differences 
    between consecutive rows.
    """
    # Check if data has a datetime index or timestamp column
    if timestamp_column:
        if timestamp_column not in data.columns:
            raise ValueError(f"Column '{timestamp_column}' not found in data.")
        time_diffs = data[timestamp_column].sort_values().diff().dropna()
    elif isinstance(data.index, pd.DatetimeIndex):
        time_diffs = data.index.to_series().diff().dropna()
    else:
        raise ValueError("Data must have a datetime index or a specified timestamp column.")
    
    # Find the most common time difference
    most_common_diff = time_diffs.mode()[0]
    
    # Determine sample time based on the most common interval
    if most_common_diff == pd.Timedelta(minutes=15):
        return '15min'
    elif most_common_diff == pd.Timedelta(minutes=5):
        return '5min'
    elif most_common_diff == pd.Timedelta(hours=1):
        return 'hourly'
    else:
        raise ValueError("Unrecognized sample time. Supported intervals are 5 minutes, 15 minutes, and hourly.")
    
def imputing_missing(data:pd.DataFrame, column:str, imputation_method:str = "ffill"):
    data_copy = data.copy()
    validate_dataframe(data_copy, required_columns=[column])
    if imputation_method == 'ffill':
        data_copy[column].fillna(method='ffill', inplace=True)
    elif imputation_method == 'bfill':
        data_copy[column].fillna(method='bfill', inplace=True)
    elif imputation_method == 'interpolate':
        data_copy[column].interpolate(inplace=True)
    return data_copy
    
def check_and_handle_missing_data(data: pd.DataFrame, columns, trendfill: bool, imputation_method: str = "ffill", drop_streaks: bool = True):
    """
    Check for missing data in the specified columns, evaluate continuity, 
    and handle based on the duration of missing data.

    Parameters:
    - data (pd.DataFrame): The DataFrame to check and handle missing data in.
    - columns (str or list): The column(s) to check for missing data.
    - trendfill (bool): Whether to fill using trend decomposition.
    - imputation_method (str): The method to use for imputation (e.g., 'ffill', 'bfill', 'interpolate').
    - drop_streaks (bool): If True, drops rows with missing streaks exceeding the threshold. Default is True.

    Returns:
    - pd.DataFrame: The modified DataFrame with missing data handled.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        try:
            validate_dataframe(data, required_columns=[column])
        except ValueError as e:
            raise ValueError(f"Error in processing the dataframe: ensure you have column {column}") from e

        sample_time = _infer_sample_time(data=data, timestamp_column="timestamps")
        
        # Determine threshold for continuous missing data based on sample time
        if sample_time == '15min':
            threshold = 96
        elif sample_time == '5min':
            threshold = 288
        elif sample_time == 'hourly':
            threshold = 24
        else:
            raise ValueError("sample_time must be one of '15min', '5min', or 'hourly'")
        
        # Identify missing data
        missing_indices = data[data[column].isna()].index
        
        # Check for continuity of missing indices
        missing_streaks = []
        if not missing_indices.empty:
            current_streak = [missing_indices[0]]
            for i in range(1, len(missing_indices)):
                if missing_indices[i] == missing_indices[i - 1] + 1:
                    current_streak.append(missing_indices[i])
                else:
                    missing_streaks.append(current_streak)
                    current_streak = [missing_indices[i]]
            missing_streaks.append(current_streak)

        # Process each missing streak
        for streak in missing_streaks:
            if len(streak) >= threshold:
                if drop_streaks:
                    # Drop rows if missing streak is longer than threshold
                    data.drop(streak, inplace=True)
            # else:
                # Handle missing values in the streak using imputation
            if trendfill:
                data[column] = data[column].interpolate(method="linear")
                result = seasonal_decompose(data[column], model='additive', period=threshold)
                trend_values = result.trend.loc[streak]
                data.loc[streak, column] = trend_values
            else:
                imputed_data = imputing_missing(data=data, column=column, imputation_method=imputation_method)
                data.loc[streak, column] = imputed_data.loc[streak, column]

    return data

##-------------Plotting---------------------
def plottingfit(data:pd.Series):
    """
    Plot the fitted normal distribution, and KDE for the specified column.
    
    Parameters:
    - data (pd.Series): The Series containing the data to be plotted.
    """
    # Calculate mu and std from the input series
    series = data.dropna()  # Remove NaN values to prevent issues with fitting
    mu, std = norm.fit(series)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the fitted normal distribution
    x = np.linspace(mu - 4*std, mu + 4*std, 1000000)
    y = norm.pdf(x, mu, std)
    ax1.plot(x, y, label='Fitted Distribution', color='k')

    # Plot the KDE of the data column
    sns.kdeplot(series, color='red', ax=ax1, label='KDE', linewidth=3)

    # Add labels and skewness annotation
    ax1.set_ylabel('Density')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.text(0.5, 0.1, f'Skew: {series.skew():.2f}', transform=ax1.transAxes,
             horizontalalignment='center', color='red', weight='bold', fontsize=14)

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    # Show the plot
    plt.show()

def qqplot(data, dist, maxi, column):
    """
    Generate a QQ plot for the given distribution and data column and calculate the R^2 score.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to be plotted.
    - dist (str): The name of the distribution to fit.
    - maxi (float): The current maximum R^2 score.
    - column (str): The column name to be analyzed.
    
    Returns:
    - maxi (float): Updated maximum R^2 score.
    - s (str): The distribution with the best fit.
    """
    # Create a probability plot
    (osm, osr), (slope, intercept, r) = probplot(data[column], dist=dist, plot=None)
    plt.figure(figsize=(10, 6))
    plt.scatter(osm, osr, s=20, label='Data')  # Scatter plot of the ordered data
    plt.plot(osm, slope * osm + intercept, 'r-', label='Fit')  # Line fit

    # Customize the plot
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values')
    plt.title(f'Q-Q plot for {dist} distribution')
    plt.legend()
    plt.show()

    # Update the maximum R^2 score and corresponding distribution
    if maxi < (r**2):
        maxi = (r**2)
        s = dist
    else:
        maxi=maxi

    print(f"R^2 score for {dist}: {r**2:.4f}")
    return maxi, s

def find_best_fit_distribution(data, column):
    """
    Iterate over all continuous distributions in scipy.stats and plot QQ plots,
    identifying the distribution with the best R^2 score.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to be analyzed.
    - column (str): The column name to be analyzed.
    """
    continuous_distributions = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
    maxi = 0
    best_fit = None

    for dist in continuous_distributions:
        try:
            maxi, s = qqplot(data, dist, maxi, column)
            if s:
                best_fit = s
                print(f"Current best fit: {best_fit}")
        except Exception as e:
            print(f"Exception for distribution {dist}: {e}")
    
    print(f"The best fit distribution is {best_fit} with an R^2 score of {maxi:.4f}")

def box_plotting(data_x, data_y, data_hue, dataframe, title, xlabel, ylabel, legend_notreq=True, figgsize=(10, 5), verticalline=False):
    """
    Create a customized box plot using seaborn and matplotlib.

    Parameters:
    - data_x (str): Column name for the x-axis.
    - data_y (str): Column name for the y-axis.
    - data_hue (str): Column name for the hue dimension (optional).
    - dataframe (pd.DataFrame): DataFrame containing the data to plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - legend_notreq (bool): If True, suppresses the legend. Default is True.
    - figgsize (tuple): Size of the figure. Default is (10, 5).
    - verticalline (bool): If True, adds vertical lines between unique x values. Default is False.

    Returns:
    - None: Displays the box plot.
    """
    plt.figure(figsize=figgsize)
    sns.boxplot(
        x=data_x,
        y=data_y,
        data=dataframe,
        hue=data_hue,
        notch=True,
        flierprops={"marker": "x"},
        medianprops={"color": "r", "linewidth": 2}
    )

    if legend_notreq:
        plt.legend([], [], frameon=False)

    if verticalline:
        unique_x = sorted(dataframe[data_x].unique())
        for xval in unique_x:
            plt.axvline(x=xval - 0.5, color='grey', linestyle='--', linewidth=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()