# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # For better table formatting in terminal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the Excel file
file_path = "Quantathon_Data_2025_Edited.xlsx"  # Ensure this file is in the same directory as the script
xls = pd.ExcelFile(file_path)  # Load the Excel file

# Load the first sheet into a DataFrame
df = pd.read_excel(xls, sheet_name='Sheet1')

# Rename columns for clarity
df.columns = ['Date_S&P', 'S&P500', 'Bond_Rate', 'Date_Prob', 'PrDec', 'PrInc']

# Convert date columns to datetime format
df['Date_S&P'] = pd.to_datetime(df['Date_S&P'], errors='coerce')
df['Date_Prob'] = pd.to_datetime(df['Date_Prob'], errors='coerce')

# Sort Date_S&P in ascending order (earliest date first)
df = df.sort_values(by='Date_S&P', ascending=True)

                    
# Compute log returns for S&P 500 allows it to be time additive
df['Log_Returns'] = np.log(df['S&P500'] / df['S&P500'].shift(1))

# Compute daily percent returns (in decimal form)
#df['Percent_Returns'] = df['S&P500'].pct_change()

# Compute 10-day rolling volatility (market uncertainty)
'''
Take the last 10 log return values.
Compute the standard deviation of those values.
Assign that standard deviation as the "Volatility" for the current day.
Move to the next day and repeat the process.
High number -> Large daily price changes
Low number -> Small daily price changes
'''
df['Volatility_10d'] = df['Log_Returns'].rolling(window=10).std()

# Compute 3-day momentum to detect short-term market trends
'''
+ momentum -> prices increasing
- momentum -> prices decresing
'''
df['Momentum_3d'] = (df['S&P500'] - df['S&P500'].shift(3)) / df['S&P500'].shift(3)


# Filter data where PrDec and PrInc are known (after Jan 26, 2022)
train_df = df[df['Date_S&P'] >= '2022-01-26'].dropna(subset=['PrDec', 'PrInc'])

# Define features (X) and target variables (y)

# Model will learn from Volatility_10d', 'Momentum_3d', 'Log_Returns'
features = ['Volatility_10d', 'Momentum_3d', 'Log_Returns']
X_train = train_df[features].dropna()
y_train_PrDec = train_df.loc[X_train.index, 'PrDec'] # values from 2022 - 2024
y_train_PrInc = train_df.loc[X_train.index, 'PrInc'] # values from 2022 - 2024

# Train separate Random Forest models for PrDec & PrInc
# 100 random trees and a fixed randomness of 42 to replicate data
rf_PrDec = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_PrInc = RandomForestRegressor(n_estimators=100, random_state=42)

#Train the models based off known data
rf_PrDec.fit(X_train, y_train_PrDec)
rf_PrInc.fit(X_train, y_train_PrInc)

# Predict PrDec & PrInc for missing data (before 2022)

# Finds all rows in df before January 26, 2022, where PrDec and PrInc are missing.
# Removes rows that have NaN in features (Volatility_10d, Momentum_3d, Log_Returns)
missing_df = df[df['Date_S&P'] < '2022-01-26'].dropna(subset=features)

# checks if there are any rows with missing PrDec and PrInc and calculates them
# Selects only the feature columns (Volatility_10d, Momentum_3d, Log_Returns) for rows before 2022.
# Because the machine learning model was trained to predict PrDec and PrInc based on these features.
if not missing_df.empty:
    X_missing = missing_df[features]
    df.loc[X_missing.index, 'PrDec'] = rf_PrDec.predict(X_missing)
    df.loc[X_missing.index, 'PrInc'] = rf_PrInc.predict(X_missing)

    # Ensure probabilities remain valid
    df['PrDec'] = np.clip(df['PrDec'], 0, 1)
    df['PrInc'] = np.clip(df['PrInc'], 0, 1 - df['PrDec'])


'''
 Bear Market:
    <=-20% from its peak to trough from recent high
 Bull Market:
    >=20% from its peak to trough from recent low 
 Static Market:
    if neither met
'''
    # Compute cumulative maximum and cumulative minimum of S&P500 prices
df['CumMax'] = df['S&P500'].cummax()  # Highest price seen up to that day
df['CumMin'] = df['S&P500'].cummin()  # Lowest price seen up to that day

# Initialize Market_State with a default value (e.g. 'Static')
df['Market_State'] = 'Static'

# Classify Bear Market: if current price <= 80% of cumulative maximum
df.loc[df['S&P500'] <= 0.8 * df['CumMax'], 'Market_State'] = 'Bear'

# Classify Bull Market: if current price >= 120% of cumulative minimum
df.loc[df['S&P500'] >= 1.2 * df['CumMin'], 'Market_State'] = 'Bull'

# Drop helper columns no longer needed
df = df.drop(columns=['CumMax', 'CumMin'])

# Filter data for the specific date range (2019-2022)
start_date = '2019-01-01'
end_date = '2022-12-31'
df = df[(df['Date_S&P'] >= start_date) & (df['Date_S&P'] <= end_date)]

# Display processed data in a table format
print("\n Sample Processed Data:")
print(tabulate(df.head(20), headers='keys', tablefmt='grid'))

# Save the cleaned data
df.to_excel("Processed_Quantathon_Data_2019_2022.xlsx", index=False)

# Create a function to predict market state using a more robust approach
def predict_market_state(df, prediction_window=3):
    """
    Extremely aggressive market state prediction with enhanced risk management
    """
    result_df = df.copy()
    result_df['Predicted_Market_State'] = 'Bull'  # Start bullish by default
    
    # RISK INDICATORS
    result_df['SP_1day_Return'] = result_df['S&P500'].pct_change(periods=1)
    result_df['SP_3day_Return'] = result_df['S&P500'].pct_change(periods=3)
    result_df['SP_5day_Return'] = result_df['S&P500'].pct_change(periods=5)
    
    # Volatility measures
    result_df['Volatility'] = result_df['Log_Returns'].rolling(window=10).std()
    result_df['Volatility_Change'] = result_df['Volatility'].pct_change(periods=3)
    
    # EARLY WARNING SYSTEM
    bear_conditions = (
        # Probability signals
        (result_df['PrDec'] > 0.4) |  # Moderate decline probability
        
        # Trend breaks
        (result_df['SP_1day_Return'] < -0.007) |  # Sharp daily drop
        (result_df['SP_3day_Return'] < -0.015) |  # Sustained decline
        
        # Volatility spikes
        (result_df['Volatility'] > 1.2 * result_df['Volatility'].rolling(window=20).mean())
    )
    
    # Go to bonds early
    result_df.loc[bear_conditions, 'Predicted_Market_State'] = 'Bear'
    
    # AGGRESSIVE RECOVERY DETECTION
    bull_conditions = (
        # Strong recovery signals
        (result_df['SP_3day_Return'] > 0.015) |  # Strong 3-day rally
        ((result_df['PrInc'] > 0.6) & (result_df['SP_1day_Return'] > 0))  # High increase probability with positive return
    )
    
    result_df.loc[bull_conditions, 'Predicted_Market_State'] = 'Bull'
    
    return result_df

# Implement a more efficient backtest strategy
def backtest_strategy(df, initial_investment=10000, start_date='2019-01-01', end_date='2022-12-31'):
    """
    Modified backtesting with more reasonable leverage and returns
    """
    # Strictly filter the date range first
    backtest_df = df[(df['Date_S&P'] >= start_date) & (df['Date_S&P'] <= end_date)].copy()
    
    # Ensure numeric columns
    numeric_cols = ['S&P500', 'Bond_Rate', 'PrDec', 'PrInc']
    for col in numeric_cols:
        backtest_df[col] = pd.to_numeric(backtest_df[col], errors='coerce')
    
    # Calculate returns
    backtest_df['S&P_Daily_Return'] = backtest_df['S&P500'].pct_change().fillna(0)
    
    # Completely revamped bond return calculation
    # Convert annual rate to daily rate and account for compounding
    backtest_df['Bond_Daily_Return'] = backtest_df['Bond_Rate'].apply(lambda x: (1 + x/100)**(1/252) - 1)
    
    # Calculate additional return metrics needed for strategy
    backtest_df['SP_1day_Return'] = backtest_df['S&P500'].pct_change(periods=1)
    backtest_df['SP_3day_Return'] = backtest_df['S&P500'].pct_change(periods=3)
    
    # Initialize signal
    backtest_df['Signal'] = 1.0
    
    if 'Predicted_Market_State' in backtest_df.columns:
        # Position management logic remains the same
        backtest_df['Signal'] = 1.0
        
        bear_mask = (
            (backtest_df['PrDec'] > 0.5) | 
            (backtest_df['SP_1day_Return'] < -0.02) |
            (backtest_df['Predicted_Market_State'] == 'Bear')
        )
        backtest_df.loc[bear_mask, 'Signal'] = 0.0
        
        strong_bull = (
            (backtest_df['PrInc'] > 0.7) & 
            (backtest_df['SP_3day_Return'] > 0.02) & 
            (backtest_df['Predicted_Market_State'] == 'Bull')
        )
        backtest_df.loc[strong_bull, 'Signal'] = 1.3
        
        backtest_df['Confidence'] = 1.0 - backtest_df['PrDec']
        backtest_df.loc[backtest_df['Signal'] > 0, 'Signal'] *= backtest_df.loc[backtest_df['Signal'] > 0, 'Confidence']
    
    # Calculate portfolio values
    portfolio_values = np.zeros(len(backtest_df))
    portfolio_values[0] = initial_investment
    
    # Completely revamped bond benchmark calculation
    bond_values = np.zeros(len(backtest_df))
    bond_values[0] = initial_investment
    
    # Calculate both portfolio and bond values using daily compounding
    for i in range(1, len(backtest_df)):
        sp_return = backtest_df['S&P_Daily_Return'].iloc[i]
        bond_return = backtest_df['Bond_Daily_Return'].iloc[i]
        signal = backtest_df['Signal'].iloc[i]
        
        if signal > 1.0:
            borrowing_cost = (signal - 1.0) * (bond_return + 0.02/365)
            portfolio_return = (signal * sp_return) - borrowing_cost + ((1 - signal) * bond_return)
        else:
            portfolio_return = (signal * sp_return) + ((1 - signal) * bond_return)
            
        portfolio_values[i] = portfolio_values[i-1] * (1 + portfolio_return)
        bond_values[i] = bond_values[i-1] * (1 + bond_return)
    
    backtest_df['Portfolio_Value'] = portfolio_values
    
    # Calculate benchmark values
    initial_sp_units = initial_investment / backtest_df['S&P500'].iloc[0]
    backtest_df['Benchmark_S&P'] = backtest_df['S&P500'] * initial_sp_units
    backtest_df['Benchmark_Bond'] = bond_values
    
    return backtest_df

# Apply the prediction function with improved parameters
df = predict_market_state(df, prediction_window=3)

# Print a sample of the predicted states to verify
print("\nSample of Predicted Market States:")
sample_cols = ['Date_S&P', 'S&P500', 'PrDec', 'PrInc', 'Market_State', 'Predicted_Market_State']
print(tabulate(df[sample_cols].sample(10), headers='keys', tablefmt='grid'))

# After running backtest_strategy, add visualization code
def plot_strategy_performance(backtest_results):
    """
    Plot the performance in actual dollar values
    """
    plt.figure(figsize=(15, 8))
    
    # Filter the data to ensure we only plot within our date range
    mask = (backtest_results['Date_S&P'] >= '2019-01-01') & (backtest_results['Date_S&P'] <= '2022-12-31')
    plot_data = backtest_results[mask]
    
    # Plot actual dollar values instead of normalized values
    plt.plot(plot_data['Date_S&P'], plot_data['Benchmark_S&P'], 
             label='S&P 500', color='orange', linewidth=2)
    plt.plot(plot_data['Date_S&P'], plot_data['Benchmark_Bond'], 
             label='Bonds', color='green', linewidth=2)
    plt.plot(plot_data['Date_S&P'], plot_data['Portfolio_Value'], 
             label='Strategy', color='blue', linewidth=2.5)
    
    # Set explicit x-axis limits
    plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31'))
    
    # Format y-axis to show dollar values
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.title('Investment Strategy Performance (Starting with $10,000)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    
    # Calculate and print metrics using actual values
    total_return = ((plot_data['Portfolio_Value'].iloc[-1] / plot_data['Portfolio_Value'].iloc[0]) - 1) * 100
    sp_return = ((plot_data['Benchmark_S&P'].iloc[-1] / plot_data['Benchmark_S&P'].iloc[0]) - 1) * 100
    bond_return = ((plot_data['Benchmark_Bond'].iloc[-1] / plot_data['Benchmark_Bond'].iloc[0]) - 1) * 100
    
    print("\nPerformance Metrics:")
    print(f"Strategy Total Return: {total_return:.1f}%")
    print(f"S&P 500 Total Return: {sp_return:.1f}%")
    print(f"Bonds Total Return: {bond_return:.1f}%")
    
    # Calculate and print risk metrics with explicit fill_method=None
    strategy_vol = plot_data['Portfolio_Value'].pct_change(fill_method=None).std() * np.sqrt(252) * 100
    sp_vol = plot_data['Benchmark_S&P'].pct_change(fill_method=None).std() * np.sqrt(252) * 100
    
    print("\nRisk Metrics:")
    print(f"Strategy Volatility: {strategy_vol:.1f}%")
    print(f"S&P 500 Volatility: {sp_vol:.1f}%")
    print(f"Sharpe Ratio: {(total_return - bond_return)/(strategy_vol):.2f}")
    
    # Print strategy behavior
    print("\nStrategy Diagnostics:")
    print(f"Number of trades: {(plot_data['Signal'].diff() != 0).sum()}")
    print(f"Average leverage: {plot_data['Signal'].mean():.2f}x")
    
    plt.show()

# Run the backtest with fixed date range
backtest_results = backtest_strategy(df, initial_investment=10000, 
                                   start_date='2019-01-01', 
                                   end_date='2022-12-31')

# Plot the results
plot_strategy_performance(backtest_results)

# Save the results
backtest_results.to_excel("Strategy_Backtest_Results.xlsx", index=False)
