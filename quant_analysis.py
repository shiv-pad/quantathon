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
    

# Display processed data in a table format
print("\n Sample Processed Data:")
print(tabulate(df.head(12), headers='keys', tablefmt='grid'))

# Save the cleaned data
df.to_excel("Processed_Quantathon_Data_Edited.xlsx", index=False)
