# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # For better table formatting in terminal

# Load the Excel file
file_path = "Quantathon_Data_2025_Edited.xlsx"  # Ensure this file is in the same directory as the script
xls = pd.ExcelFile(file_path)  # Load the Excel file

# Load the first sheet into a DataFrame
df = pd.read_excel(xls, sheet_name='Sheet1')

# Print original column names to verify structure
print("\n Original column names:", df.columns)

# Rename columns for clarity
df.columns = ['Date_S&P', 'S&P500', 'Bond_Rate', 'Date_Prob', 'PrDec', 'PrInc']

# Convert date columns to datetime format
df['Date_S&P'] = pd.to_datetime(df['Date_S&P'], errors='coerce')
df['Date_Prob'] = pd.to_datetime(df['Date_Prob'], errors='coerce')


# Ensure Date_S&P is the index before interpolation
df = df.set_index('Date_S&P')


# Interpolate missing PrDec and PrInc using time-based interpolation
df['PrDec'] = df['PrDec'].interpolate(method='time')
df['PrInc'] = df['PrInc'].interpolate(method='time')

# Reset index after interpolation
df = df.reset_index()

# Sort Date_S&P in descending order (latest date first)
df = df.sort_values(by='Date_S&P', ascending=False)
                    
# Compute log returns for S&P 500
df['Log_Returns'] = np.log(df['S&P500'] / df['S&P500'].shift(1))

# Compute daily percent returns (in decimal form)
#df['Percent_Returns'] = df['S&P500'].pct_change()

# Compute 10-day rolling volatility (market uncertainty)
'''
Take the last 10 log return values.
Compute the standard deviation of those values.
Assign that standard deviation as the "Volatility" for the current day.
Move to the next day and repeat the process.
High number - Large daily price changes
Low number - Small daily price changes
'''
df['Volatility_10d'] = df['Log_Returns'].rolling(window=10).std()

# Compute 3-day momentum to detect short-term market trends
df['Momentum_3d'] = (df['S&P500'] - df['S&P500'].shift(3)) / df['S&P500'].shift(3)

# Ensure PrDec & PrInc sum to at most 1
df['PrDec'] = np.clip(df['PrDec'], 0, 1)
df['PrInc'] = np.clip(df['PrInc'], 0, 1 - df['PrDec'])

# Display processed data in a table format
print("\n Sample Processed Data:")
print(tabulate(df.head(12), headers='keys', tablefmt='grid'))

# Save the cleaned data
df.to_excel("Processed_Quantathon_Data_Edited.xlsx", index=False)
