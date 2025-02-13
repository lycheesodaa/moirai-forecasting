import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series_splits(data, title="Time Series Split Visualization",
                            steps_before=250, steps_after=60):
    """
    Plot a time series with train/validation/test splits in different colors,
    focusing on a window around the test split.

    Parameters:
    data : array-like
        The time series data to be split and plotted
    title : str
        Title for the plot
    steps_before : int
        Number of time steps to show before the test split
    steps_after : int
        Number of time steps to show after the test split begins
    """
    # Calculate split indices
    n = len(data)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    test_start = train_size + val_size

    # Calculate window boundaries
    window_start = max(0, test_start - steps_before)
    window_end = min(n, test_start + steps_after)

    # Create time axis
    time = np.arange(n)

    # Create the plot
    fig = plt.figure(figsize=(15, 6))

    # Plot training data (blue)
    mask = (time >= window_start) & (time < train_size)
    plt.plot(time[mask], data[mask], 'b-', label='Training (60%)')

    # Plot validation data (green)
    mask = (time >= train_size) & (time < test_start) & (time >= window_start)
    plt.plot(time[mask], data[mask], 'b-', label='Validation (20%)')

    # Plot test data (red)
    mask = (time >= test_start) & (time < window_end)
    plt.plot(time[mask], data[mask], 'r-', label='Test (20%)')

    # Add vertical line to show split point
    plt.axvline(x=test_start, color='gray', linestyle='--', alpha=0.5,
                label='Test Split Start')

    # Customize the plot
    plt.title('TSLA close prices')
    plt.xlabel('Time Step')
    plt.ylabel('Close price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set the x-axis limits to our window of interest
    plt.xlim(window_start, window_end)

    fig.savefig(f'test.png', bbox_inches='tight')

    return plt.gcf()


# Example usage:
# Generate sample data
# np.random.seed(42)
# n_points = 1000
# t = np.linspace(0, 4 * np.pi, n_points)
# sample_data = np.sin(t) + 0.2 * np.random.randn(n_points)
df = pd.read_csv('day_average_headlines/TSLA.csv')

# Create and show the plot
plot_time_series_splits(df['close'].to_numpy(), "Time Series with Train/Validation/Test Splits")
plt.show()