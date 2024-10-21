import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.spatial import KDTree

# Function to find local minima and maxima with an increased timeframe
def find_local_extrema(prices, dates, window=7):
    buy_points = []
    sell_points = []
    
    for i in range(window, len(prices) - window):
        # Find local minima
        if all(prices[i] < prices[i - j] for j in range(1, window + 1)) and all(prices[i] < prices[i + j] for j in range(1, window + 1)):
            buy_points.append((prices[i], i))

        # Find local maxima
        if all(prices[i] > prices[i - j] for j in range(1, window + 1)) and all(prices[i] > prices[i + j] for j in range(1, window + 1)):
            sell_points.append((prices[i], i))

    return buy_points, sell_points

# Function to highlight top local gains with a dynamic timeframe and slope-based filtering
def get_top_local_gains(prices, dates, buy_points, sell_points, min_gain=3):
    gains = []
    
    # Calculate gains between each pair of buy and sell points
    for buy_price, buy_index in buy_points:
        for sell_price, sell_index in sell_points:
            if sell_index > buy_index:  # Ensure sell comes after buy
                # Calculate percentage gain
                gain = (sell_price - buy_price) / buy_price * 100
                
                # Check if gain is above the defined threshold
                if gain > min_gain:
                    time_diff = (dates[sell_index] - dates[buy_index]).days
                    gain_rate = gain / time_diff  # Calculate gain rate
                    gains.append((gain, buy_price, sell_price, buy_index, sell_index, time_diff, gain_rate))
    
    # Sort by gain and select top 5
    top_gains = sorted(gains, key=lambda x: (x[0], x[6]), reverse=True)

    # Filter out overlapping segments
    non_overlapping_gains = []
    used_indices = set()

    for gain_info in top_gains:
        gain, buy_price, sell_price, buy_index, sell_index, time_diff, gain_rate = gain_info
        if not any(index in used_indices for index in range(buy_index, sell_index + 1)):
            non_overlapping_gains.append(gain_info)
            used_indices.update(range(buy_index, sell_index + 1))
        
        if len(non_overlapping_gains) >= 5:  # Stop if we have 5 top gains
            break

    return non_overlapping_gains

# Anomaly detection based on price change percentage
def find_anomalies(prices, percentage_threshold):
    anomalies = []
    for i in range(1, len(prices)):
        price_change = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
        if abs(price_change) > percentage_threshold:
            anomalies.append((prices[i], i))
    return anomalies

# Closest Pair of Points for Anomaly Detection
def closest_pair(prices, dates, threshold=0.05):
    # Convert dates to UNIX timestamp (seconds since epoch) for comparison
    date_timestamps = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  # Convert to seconds
    data = np.column_stack((date_timestamps, prices))  # Use timestamps and prices for KDTree
    tree = KDTree(data)
    
    anomalies = []
    
    for i in range(len(data)):
        # Find close points within the threshold
        indices = tree.query_ball_point(data[i], r=threshold)  
        for j in indices:
            if i != j:  # Avoid pairing the point with itself
                price_diff = abs(prices[i] - prices[j])
                if price_diff / prices[i] > 0.05:  # Define your anomaly threshold as a fraction of price
                    anomalies.append((prices[i], prices[j], i, j, price_diff))

    return anomalies

# Kadane's Algorithm for Maximum Gain
def kadane(prices):
    max_current = max_global = prices[0]
    start = end = s = 0
    
    for i in range(1, len(prices)):
        if prices[i] > max_current + prices[i]:
            max_current = prices[i]
            s = i
        else:
            max_current += prices[i]
        
        if max_current > max_global:
            max_global = max_current
            start = s
            end = i
            
    return max_global, start, end

# Main Code to Analyze Stock Data
ticker = input("Enter the stock ticker (e.g., QQQ, SPY): ").upper()
stock_data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Reset the index to get the Date as a column
stock_data.reset_index(inplace=True)
prices = stock_data['Close'].values
dates = stock_data['Date']  # Store the dates for time comparisons

# Find local minima and maxima with an increased window
local_buy_points, local_sell_points = find_local_extrema(prices, dates, window=7)  # Increased window size to 7 days

# Calculate local gains and get top 5 non-overlapping
min_gain_threshold = 3  # Minimum gain percentage
top_local_gains = get_top_local_gains(prices, dates, local_buy_points, local_sell_points, min_gain=min_gain_threshold)

# Anomaly detection using a lower percentage change threshold
percentage_threshold = 4  # Lowered percentage threshold for anomalies
anomalies = find_anomalies(prices, percentage_threshold)

# Use closest pair algorithm for anomaly detection
closest_anomalies = closest_pair(prices, dates, threshold=0.05)

# Prepare for plotting
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], prices, label=f'{ticker} Close Price', color='blue')

# Initialize a set to keep track of legend labels for top local gains
top_gain_labels = set()

# Plot local buy points with larger markers
for price, index in local_buy_points:
    plt.plot(stock_data['Date'][index], price, 'go', markersize=10, label='Local Buy Point' if 'Local Buy Point' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot local sell points with larger markers
for price, index in local_sell_points:
    plt.plot(stock_data['Date'][index], price, 'ro', markersize=10, label='Local Sell Point' if 'Local Sell Point' not in plt.gca().get_legend_handles_labels()[1] else "")

# Highlight the segments of the top local gains with lighter shades of yellow
for period_number, (gain, buy_price, sell_price, buy_index, sell_index, time_diff, gain_rate) in enumerate(top_local_gains, start=1):
    # Normalize the gain for color intensity (0 to 1) based on gain percentage
    norm_gain = (gain - min_gain_threshold) / (10 - min_gain_threshold)  # Adjust the maximum gain for normalization
    norm_gain = np.clip(norm_gain, 0, 1)  # Clipping values to [0, 1]
    
    # Define the lighter shade of yellow based on normalized gain
    r = 1  # Red component (1 for yellow)
    g = 1  # Green component (1 for yellow)
    b = 0.5 + 0.5 * (1 - norm_gain)  # Slightly increase blue component to make yellow lighter
    
    # Create the color with varying lighter yellow shades
    color = (r, g, b)  # Removed alpha component
    
    plt.fill_between(stock_data['Date'][buy_index:sell_index + 1], 
                     prices[buy_index:sell_index + 1], 
                     color=color)
    
    # Add entry for the legend based on gain
    gain_label = f'Top Gain: {gain:.2f}%'
    top_gain_labels.add(gain_label)
    
    # Label the segments with percentage gain
    mid_index = (buy_index + sell_index) // 2
    plt.text(stock_data['Date'][mid_index], (buy_price + sell_price) / 2 + 1, f'{gain:.2f}%',  # Place the label above the midpoint
             fontsize=12, ha='center', color='black')

# Find and plot anomalies from the percentage change method
for price, index in anomalies:
    plt.plot(stock_data['Date'][index], price, 'purple', marker='o', markersize=5, label='Anomaly' if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")

# Find and plot anomalies from the closest pair method
for price1, price2, index1, index2, price_diff in closest_anomalies:
    plt.plot(stock_data['Date'][index1], price1, 'orange', marker='x', markersize=10, label='Closest Anomaly' if 'Closest Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(stock_data['Date'][index2], price2, 'orange', marker='x', markersize=10)

# Calculate daily percentage changes for Kadane's Algorithm
percentage_changes = np.diff(prices) / prices[:-1] * 100
max_gain, max_start_index, max_end_index = kadane(percentage_changes)

# Plot the buy and sell points for the maximum gain
buy_price = prices[max_start_index]
sell_price = prices[max_end_index + 1]  # Because of np.diff, sell point is the next price
plt.plot(stock_data['Date'][max_start_index], buy_price, 'yo', markersize=10, label='Max Gain Buy Point')
plt.plot(stock_data['Date'][max_end_index + 1], sell_price, 'yo', markersize=10, label='Max Gain Sell Point')

# Final plot adjustments
plt.title(f'{ticker} Stock Prices with Local Buy/Sell Points and Anomalies')
plt.xlabel('Date')
plt.ylabel('Price')

# Create a custom legend
handles, labels = plt.gca().get_legend_handles_labels()

# Include top gains in the legend
for gain_label in top_gain_labels:
    handles.append(plt.Line2D([0], [0], color='yellow', lw=4))  # Adding a dummy line for color
    labels.append(gain_label)

# Move the legend to the bottom right corner
plt.legend(handles, labels, loc='lower right')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output percentage gain information for local gains
if top_local_gains:
    for i, (gain, buy_price, sell_price, buy_index, sell_index, time_diff, gain_rate) in enumerate(top_local_gains):
        print(f'Top Local Gain {i + 1}: {gain:.2f}% from {buy_price:.2f} to {sell_price:.2f} over {time_diff} days with gain rate: {gain_rate:.2f}')
else:
    print('No local gains found.')

# Print the maximum gain using Kadane's Algorithm
print(f"Maximum Gain using Kadane's Algorithm: {max_gain:.2f}% from {buy_price:.2f} to {sell_price:.2f} between {max_start_index} and {max_end_index + 1}")
