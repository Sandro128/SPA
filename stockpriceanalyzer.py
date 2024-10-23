import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.spatial import KDTree

# Constants
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 7
MIN_GAIN_THRESHOLD = 3  # Minimum percentage gain
PERCENTAGE_THRESHOLD = 4  # Anomaly threshold as a percentage
CLOSEST_PAIR_THRESHOLD = 0.05  # Closest pair distance threshold
TOP_GAIN_LIMIT = 5  # Max number of top gains to display

# Function to find local minima and maxima
def find_local_extrema(prices, dates, window=WINDOW_SIZE):
    buy_points, sell_points = [], []

    for i in range(window, len(prices) - window):
        if all(prices[i] < prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] < prices[i + j] for j in range(1, window + 1)):
            buy_points.append((prices[i], i))

        if all(prices[i] > prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] > prices[i + j] for j in range(1, window + 1)):
            sell_points.append((prices[i], i))

    return buy_points, sell_points

# Function to highlight top local gains
def get_top_local_gains(prices, dates, buy_points, sell_points, min_gain=MIN_GAIN_THRESHOLD):
    gains = []

    for buy_price, buy_index in buy_points:
        for sell_price, sell_index in sell_points:
            if sell_index > buy_index:
                gain = (sell_price - buy_price) / buy_price * 100
                if gain > min_gain:
                    time_diff = (dates[sell_index] - dates[buy_index]).days
                    gain_rate = gain / time_diff
                    gains.append((gain, buy_price, sell_price, buy_index, sell_index, time_diff, gain_rate))

    top_gains = sorted(gains, key=lambda x: (x[0], x[6]), reverse=True)

    non_overlapping_gains, used_indices = [], set()
    for gain_info in top_gains:
        if not any(idx in used_indices for idx in range(gain_info[3], gain_info[4] + 1)):
            non_overlapping_gains.append(gain_info)
            used_indices.update(range(gain_info[3], gain_info[4] + 1))
        if len(non_overlapping_gains) >= TOP_GAIN_LIMIT:
            break

    return non_overlapping_gains

# Anomaly detection by percentage change
def find_anomalies(prices):
    return [(prices[i], i) for i in range(1, len(prices)) if abs((prices[i] - prices[i - 1]) / prices[i - 1] * 100) > PERCENTAGE_THRESHOLD]

# Closest pair algorithm for anomaly detection
def closest_pair(prices, dates):
    date_timestamps = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    data = np.column_stack((date_timestamps, prices))
    tree = KDTree(data)

    anomalies = []
    for i in range(len(data)):
        indices = tree.query_ball_point(data[i], r=CLOSEST_PAIR_THRESHOLD)
        for j in indices:
            if i != j:
                price_diff = abs(prices[i] - prices[j])
                if price_diff / prices[i] > 0.05:
                    anomalies.append((prices[i], prices[j], i, j, price_diff))
    return anomalies

# Kadane's Algorithm
def kadane(prices):
    max_current = max_global = prices[0]
    start = end = s = 0

    for i in range(1, len(prices)):
        if prices[i] > max_current + prices[i]:
            max_current, s = prices[i], i
        else:
            max_current += prices[i]

        if max_current > max_global:
            max_global, start, end = max_current, s, i

    return max_global, start, end

# Main Program
ticker = input("Enter the stock ticker (e.g., QQQ, SPY): ").upper()
stock_data = yf.download(ticker, start=START_DATE, end=END_DATE)

stock_data.reset_index(inplace=True)
prices = stock_data['Close'].values
dates = stock_data['Date']

local_buy_points, local_sell_points = find_local_extrema(prices, dates)
top_local_gains = get_top_local_gains(prices, dates, local_buy_points, local_sell_points)
anomalies = find_anomalies(prices)
closest_anomalies = closest_pair(prices, dates)

percentage_changes = np.diff(prices) / prices[:-1] * 100
max_gain, max_start_index, max_end_index = kadane(percentage_changes)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], prices, label=f'{ticker} Close Price', color='blue')

# Plot buy and sell points
for price, index in local_buy_points:
    plt.plot(stock_data['Date'][index], price, 'go', markersize=10, label='Local Buy Point' if 'Local Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")

for price, index in local_sell_points:
    plt.plot(stock_data['Date'][index], price, 'ro', markersize=10, label='Local Sell Point' if 'Local Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")

# Highlight top local gains
top_gain_handles = []  # List to hold custom handles for the legend
for gain_info in top_local_gains:
    _, _, _, buy_idx, sell_idx, _, _ = gain_info
    fill = plt.fill_between(stock_data['Date'][buy_idx:sell_idx + 1], prices[buy_idx:sell_idx + 1], color='yellow', alpha=0.3)
    
    # Create a custom legend handle for this gain
    top_gain_handles.append(plt.Line2D([0], [0], color='yellow', lw=10, label=f'Top Gain: {gain_info[0]:.2f}%'))

# Plot anomalies
for price, index in anomalies:
    plt.plot(stock_data['Date'][index], price, 'purple', marker='o', markersize=5, label='Anomaly' if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")

for price1, price2, idx1, idx2, _ in closest_anomalies:
    plt.plot(stock_data['Date'][idx1], price1, 'orange', marker='x', markersize=10, label='Closest Anomaly' if 'Closest Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(stock_data['Date'][idx2], price2, 'orange', marker='x', markersize=10)

# Plot Kadane's max gain points
plt.plot(stock_data['Date'][max_start_index], prices[max_start_index], 'yo', markersize=10, label='Max Gain Buy Point')
plt.plot(stock_data['Date'][max_end_index + 1], prices[max_end_index + 1], 'yo', markersize=10, label='Max Gain Sell Point')

# Add custom handles for top local gains to the legend
plt.legend(handles=top_gain_handles + plt.gca().get_legend_handles_labels()[0], loc='lower right')

# Adjust plot
plt.title(f'{ticker} Stock Prices with Buy/Sell Points and Anomalies')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

