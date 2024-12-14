import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

class general:
    
    def interpolate_outliers(weights, label,show_plot = False):
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
    
        if "Weight" in label:
            for i, value in enumerate(weights_interpolated):
                if value <= 0:
                    # Find the next valid (positive) value to use
                    next_positive = next((v for v in weights_interpolated[i:] if v > 0), 1e-6)  # Use a small positive value if no valid value is found
                    weights_interpolated[i] = next_positive
        
        if show_plot:
            # Plot
            plt.figure(figsize=(10, 6))
        
            # Original data with outliers highlighted
            plt.plot(indices, weights, label='Original Data', marker='o')
            plt.scatter(outlier_indices, weights[outliers], color='red', label='Outliers', zorder=5)
        
            # Interpolated data
            plt.plot(indices, weights_interpolated, label='Interpolated Data', linestyle='--', marker='x')
        
            plt.title("Outlier Detection and Interpolation")
            plt.xlabel("Index")
            plt.ylabel(label)
            plt.legend()
            plt.grid()
            plt.show()
    
        return weights_interpolated



    
    def compute_metrics(predicted_values, actual_values):
        # Convert inputs to numpy arrays for safety
        predicted_values = np.array(predicted_values)
        actual_values = np.array(actual_values)
        
        # Calculate MSE
        mse = np.mean((predicted_values - actual_values) ** 2)
        
        # Calculate MAE
        mae = np.mean(np.abs(predicted_values - actual_values))
        
        # Calculate MAPE (handling division by zero)
        # Use np.where to avoid division by zero
        non_zero_actuals = np.where(actual_values == 0, np.nan, actual_values)
        mape = np.mean(np.abs((actual_values - predicted_values) / non_zero_actuals)) * 100
        
        # Replace NaNs with a large value or skip them as appropriate
        if np.isnan(mape):
            mape = float('inf')  # Assign infinity if MAPE is undefined
        
        # Print the results
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}%")
        
        return mse, mae, mape
    
    # Log the training and validation metrics to TensorBoard
    def log_metrics(writer, history):
        for epoch in range(len(history.history['loss'])):
            writer.add_scalar("Loss/Train", history.history['loss'][epoch], epoch)
            writer.add_scalar("Loss/Val", history.history['val_loss'][epoch], epoch)
            writer.add_scalar("MSE/Train", history.history['mse'][epoch], epoch)
            writer.add_scalar("MSE/Val", history.history['val_mse'][epoch], epoch)
            writer.add_scalar("MAE/Train", history.history['mae'][epoch], epoch)
            writer.add_scalar("MAE/Val", history.history['val_mae'][epoch], epoch)
            writer.add_scalar("MAPE/Train", history.history['mape'][epoch], epoch)
            writer.add_scalar("MAPE/Val", history.history['val_mape'][epoch], epoch)
            
    def corr_matrix(data, writer):
        # Compute the correlation matrix
        correMtr = data.corr()
        mask = np.array(correMtr)
        mask[np.tril_indices_from(mask)] = False  # the correlation matrix is symmetric

        # Prepare the weight correlations for the bar chart
        weight_corr = correMtr['PREORE_VAKI-Weight [g]'].drop('PREORE_VAKI-Weight [g]').sort_values()

        # Create a figure with 2 subplots (1x2 grid)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Plot the correlation heatmap on the first axis
        sns.set_style("white")
        sns.heatmap(correMtr, mask=mask, vmin=-1.0, vmax=1.0, square=True, annot=True, fmt=".2f", ax=ax1)
        ax1.set_title('Correlation matrix of attributes')
        
        # Plot the vertical bar chart on the second axis
        sns.barplot(y=weight_corr.index, x=weight_corr.values,hue=weight_corr.index, legend=False, ax=ax2)
        ax2.set_xlabel("Correlation with PREORE_VAKI-Weight [g]")
        ax2.set_ylabel("Features")
        ax2.set_title("Correlation of Features with PREORE_VAKI-Weight [g]")
        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        # Log the combined figure to TensorBoard
        writer.add_figure("Combined Correlation Plots", fig)
        
    def assign_labels(val_index, boundaries):
        labels = np.zeros_like(val_index, dtype=int)
        for i, (start, end) in enumerate(boundaries):
            labels[(val_index >= start) & (val_index <= end)] = i + 1
        return labels
