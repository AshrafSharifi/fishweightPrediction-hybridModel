import pandas as pd
from constants import constants
import numpy as np
import matplotlib.pyplot as plt  # Ensure correct import
from sklearn.metrics import mean_squared_error
import pickle
from general import general

# Load existing data
# root = 'data/Preore_Dataset/results/'
RNN_data = 'data/Runs/1_LSTM_WithTransform_WithTime_WithScaling_(2024-12-09_15_21_44)/'


with open(RNN_data + 'dynamic_individual_weight.pkl', 'rb') as file:
    data = pickle.load(file)
    
df_Vaki_weights_daily_mean_initial = pd.read_csv('data/Preore_Dataset/PREORE_VAKI-Weight_dailymean.csv')
# df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'] = general.interpolate_outliers(
#     df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'], 'mean_weight'
# )
df_Vaki_weights_daily_mean_initial['observed_timestamp'] = pd.to_datetime(df_Vaki_weights_daily_mean_initial['observed_timestamp'])
df_Vaki_weights_daily_mean_initial['date'] = df_Vaki_weights_daily_mean_initial['observed_timestamp'].dt.date

for i in range(3):
    start_date = pd.to_datetime(data[i]['start_date'])
    end_date = pd.to_datetime(data[i]['end_date'])
    df_data = data[i]['df']
    df_data['date'] = df_data['Entrance_timestamp'].dt.date
    
    df_Vaki_weights_daily_mean = df_Vaki_weights_daily_mean_initial[(df_Vaki_weights_daily_mean_initial['date'] >= start_date.date()) & 
                                                                    (df_Vaki_weights_daily_mean_initial['date'] <= end_date.date())].reset_index(drop=True)
    
    df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'] = general.interpolate_outliers(df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'],'Weight_dailymean [g]')
    # Group by date and filter groups with size equal to sampling rate per day
    df_data = df_data.groupby('date').filter(lambda x: len(x) == data[i]['sampling_rate_per_day'])
    predicted_weights = df_data.groupby('date')['mathematical_computed_weight'].mean().reset_index()
    predicted_weight_RNN = df_data.groupby('date')['predicted_weight_RNN'].mean().reset_index()
    Feed_ration = df_data.groupby('date')['Feed_ration'].mean().reset_index()
    
    observed_weights = []
    dates = []
    for index, row in predicted_weights.iterrows():
        temp = df_Vaki_weights_daily_mean[df_Vaki_weights_daily_mean['date'] == row['date']]
        observed_weights.append(temp['PREORE_VAKI-Weight_dailymean [g]'].iloc[0])
        dates.append(row['date'])
    
    # Convert observed weights to a NumPy array
    observed_weights = np.array(observed_weights)
    predicted_weights = np.array(predicted_weights['mathematical_computed_weight'])
    predicted_weight_RNN = np.array(predicted_weight_RNN['predicted_weight_RNN'])
    # Plot observed vs predicted weights
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, observed_weights, label='Observed Mean Weight', marker='o')
    plt.plot(dates, predicted_weights, label='Predicted Mean Weight_MathematicalMethod', linestyle='--', marker='x', color='orange')
    plt.plot(dates, predicted_weight_RNN, label='Predicted Mean Weight_RNN', linestyle='-', marker='x', color='gray')
    
    plt.title(f'Observed vs Predicted Mean Weights_TimeWindow: {i+1}')
    plt.xlabel('Date')
    plt.ylabel('Weight (g)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Calculate goodness-of-fit indices
    observed_mean = observed_weights.mean()
    predicted_mean = predicted_weights.mean()
    RNN_predicted_mean = predicted_weight_RNN.mean()
    
    
    observed_std = observed_weights.std()
    predicted_std = predicted_weights.std()
    RNN_predicted_std = predicted_weight_RNN.std()
    
    print("==============Bioenergetic model==============")
    # RMSE for the mean weights
    rmse_mean = np.sqrt(mean_squared_error([observed_mean], [predicted_mean]))
    
    # RMSE for the standard deviation
    rmse_std = np.sqrt(mean_squared_error([observed_std], [predicted_std]))

    # Display the results
    print("Goodness-of-Fit Indices:")
    print(f"RMSE (Mean Weights): {rmse_mean}")
    print(f"RMSE (Standard Deviation): {rmse_std}")
    
    print("==============RNN model==============")
    # # RMSE for the mean weights
    rmse_mean = np.sqrt(mean_squared_error([observed_mean], [RNN_predicted_mean]))
    
    # RMSE for the standard deviation
    rmse_std = np.sqrt(mean_squared_error([observed_std], [RNN_predicted_std]))
    
    # Display the results
    print("Goodness-of-Fit Indices:")
    print(f"RMSE (Mean Weights): {rmse_mean}")
    print(f"RMSE (Standard Deviation): {rmse_std}")
