import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
file_path = 'data.csv'
data = pd.read_csv(file_path, header=None, names=['Values'])
print("Original Data:")
print(data)
mean_values = np.mean(data['Values'])
data['Mean_Removed_Values'] = data['Values'] - mean_values
normalized_values = (data['Values'] - np.min(data['Values'])) / (np.max(data['Values']) - np.min(data['Values']))
data['Normalized_Values'] = normalized_values
scaler = ((data['Values'] - max(data['Values']))/max(data['Values']))
print("\nMean Removed Data:")
print(data[['Mean_Removed_Values']])
print("\nNormalized Data:")
print(data[['Normalized_Values']])
print("\nStandardized Data:")
print(scaler)
