# Project Overview
This project focuses on analyzing historical stock price data to identify local gains, detect anomalies, and maximize profit opportunities using computational algorithms. The key problems addressed include:

- **Identifying Local Gains**: Finding periods where stock prices exhibit significant increases for potential trading opportunities.
- **Anomaly Detection**: Detecting unusual price movements that could indicate market inefficiencies.
- **Maximizing Gains**: Utilizing algorithms to determine optimal buy and sell points for maximum profit.

## Goals of the Analysis
The main objectives are:

- To pinpoint the top local gains in stock prices based on a specified percentage threshold.
- To visualize the identified gains and anomalies over time.
- To employ Kadane's Algorithm to find the maximum gain in a sequence of daily percentage changes.

## Type-Specific Considerations
### Financial Datasets
**Data Source**: Historical stock price data obtained from Yahoo Finance via the yfinance library, covering the period from January 1, 2020, to January 1, 2024.

### Algorithms Used for Processing and Analysis
- **Local Extrema Detection**: Identifies local minima and maxima in stock prices using a sliding window approach for flexible timeframe analysis.

- **Gain Calculation**: Calculates percentage gains between buy and sell points, filtering out insignificant trades.

- **Kadane's Algorithm**: Finds the maximum gain from a sequence of daily percentage changes, helping identify optimal trading windows.

- **Closest Pair of Points Algorithm**: Used for anomaly detection by identifying spatially close price pairs, aiding in detecting unusual price movements.

### Considerations on Sorting
**No Need for Merge Sort**: The yfinance library provides historical stock price data in a pre-sorted format by date, eliminating the need for sorting algorithms like Merge Sort.

## Libraries Used
- **NumPy**: Facilitates efficient numerical calculations and array manipulations for large datasets.

- **Pandas**: Provides powerful data structures for handling time series data, enabling easy manipulation and analysis.

- **Matplotlib**: Used for visualizing stock price movements, buy/sell points, and anomalies.

- **yfinance**: Offers a simple interface for accessing financial data from Yahoo Finance.

- **SciPy**: Implements the KDTree for efficient spatial searching to detect anomalies based on proximity in price and time.

## Project Structure Overview
### Class Summaries
**StockAnalysis**

- **Purpose**: Orchestrates the entire analysis process.
- **Key Methods**:
  - `load_data()`: Loads historical stock price data from yfinance.
  - `identify_gains()`: Identifies local gains based on a specified percentage threshold.
  - `detect_anomalies()`: Detects unusual price movements indicating market inefficiencies.
  - `calculate_max_gain()`: Applies Kadane's Algorithm to find the maximum gain from daily percentage changes.
  - `visualize_results()`: Visualizes identified gains and anomalies using Matplotlib.

**LocalGains**

- **Purpose**: Identifies significant local gains in stock prices.
- **Key Methods**:
  - `find_local_maxima()`: Finds local maxima in stock prices using a sliding window approach.
  - `calculate_percentage_gain()`: Calculates percentage gains between buy and sell points, filtering insignificant trades.

**AnomalyDetector**

- **Purpose**: Detects anomalies in stock price movements.
- **Key Methods**:
  - `detect_anomalies()`: Identifies unusual price movements.
  - `closest_pair()`: Implements the closest pair of points algorithm for anomaly detection.

**MaxGainCalculator**

- **Purpose**: Computes maximum gain from daily percentage changes.
- **Key Methods**:
  - `apply_kadane()`: Implements Kadane's Algorithm to find the maximum gain.

## Installation
1. Install Python from python.org.
2. Install Required Libraries: Run the following commands in your terminal:
   ```bash
   pip install numpy pandas matplotlib yfinance scipy
Using the Program
Run the Script: Execute the Python script with the provided code.
Input Stock Ticker: When prompted, enter the stock ticker symbol (e.g., QQQ, SPY) in uppercase.
Constants
window: Timeframe for local extrema (7 days).
min_gain: Minimum gain percentage for top gains (3%).
percentage_threshold: Anomaly detection threshold (4%).
threshold: Distance for closest pair anomaly detection (0.05).
Verification of Code Functionality and Findings
This section outlines the verification of code functionality, detailing how the system operates in analyzing stock prices, detecting anomalies, and generating trend reports. It also discusses insights gained from the analysis, challenges encountered, and potential areas for improvement. Screenshots demonstrating the system in action will be provided later to enhance understanding.

1. Code Functionality Verification
Analyzing Stock Prices:

The program retrieves historical stock data for a specified ticker (e.g., QQQ) from January 1, 2020, to January 1, 2024. Users can easily input their desired stock ticker, and the system performs calculations to identify local minima and maxima, as well as compute gains based on user-defined thresholds.

Detecting Anomalies:

The system implements two methods for detecting anomalies:

Percentage Change Method: This method identifies significant price fluctuations and marks them as anomalies. For instance, during the early months of the COVID-19 pandemic in 2020, QQQ experienced a dramatic price drop, which the system detected as an anomaly.
Closest Pair Method: This algorithm flags close points in price changes as anomalies if the price difference exceeds a specified threshold. In 2022, multiple tech companies announced layoffs, leading to noticeable shifts in QQQ's price. These fluctuations were successfully captured by the anomaly detection algorithms.
Sample Scenarios:

Scenario 1: Analyzing stock prices for QQQ from 2020 to 2024 shows how local minima and maxima are identified. For instance, the system may detect a buy point in March 2020, followed by a sell point later in the year, illustrating a potential gain for investors.
Scenario 2: The detection of anomalies during the pandemic (2020) and tech layoffs (2022) demonstrates the system's effectiveness in flagging unusual market behavior.
Screenshots:

Screenshots will be provided to illustrate the execution of the program and its output. These images will showcase the graphical representation of stock prices, identified buy and sell points, and highlighted anomalies. They will serve as visual confirmation of the algorithms' functionality and provide clarity on how data is processed and presented.

2. Discussion of Findings
Insights Gained: The analysis revealed how external factors, such as the COVID-19 pandemic and economic conditions leading to layoffs, significantly impacted stock prices. Understanding these influences can help investors make informed decisions based on historical trends.

Challenges Faced: One challenge encountered during implementation was ensuring the accuracy of anomaly detection algorithms, particularly in distinguishing between normal price fluctuations and genuine anomalies. Adjusting thresholds required careful consideration of market conditions.

Limitations: The current system may not account for all external variables influencing stock prices, such as macroeconomic indicators or global events. This could limit the predictive power of the analysis.

Areas for Improvement: Future enhancements could include integrating additional data sources (e.g., economic indicators), refining anomaly detection algorithms, and implementing machine learning techniques to improve the accuracy of predictions. User feedback mechanisms could also be added to help tailor the analysis to specific investor needs.
