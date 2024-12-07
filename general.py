import pandas as pd
from constants import constants
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pandas as pd


class general:
    
    def interpolate_outliers(weights):
        # Detect outliers using the IQR method
        q1 = np.percentile(weights, 25)
        q3 = np.percentile(weights, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outliers = (weights < lower_bound) | (weights > upper_bound)
        outlier_indices = np.where(outliers)[0]
        
        # Replace outliers with NaN
        weights_cleaned = weights.copy()
        weights_cleaned[outliers] = np.nan
        
        # Interpolate missing values (linear interpolation)
        indices = np.arange(len(weights))
        valid_indices = ~np.isnan(weights_cleaned)
        interpolator = interp1d(indices[valid_indices], weights_cleaned[valid_indices], kind='linear', fill_value="extrapolate")
        weights_interpolated = interpolator(indices)
        
        # # Plot
        plt.figure(figsize=(10, 6))
        
        # Original data with outliers highlighted
        plt.plot(indices, weights, label='Original Data', marker='o')
        plt.scatter(outlier_indices, weights[outliers], color='red', label='Outliers', zorder=5)
        
        # Interpolated data
        plt.plot(indices, weights_interpolated, label='Interpolated Data', linestyle='--', marker='x')
        
        plt.title("Outlier Detection and Interpolation")
        plt.xlabel("Index")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid()
        plt.show()
        return weights_interpolated
    
    def compute_metrics(predicted_values,actual_values):
        # Calculate loss, MSE, MAE, and MAPE
        mse = np.mean((predicted_values - actual_values) ** 2)
        mae = np.mean(np.abs(predicted_values - actual_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

        # Print the results
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}%")
        return mse,mae,mape