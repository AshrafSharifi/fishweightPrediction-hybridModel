import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from rainbow_trout_model import *
import math


root = 'data/Preore_Dataset/'
df_timeWindow = pd.read_csv(root+'Time_window.csv')
df_data = pd.read_csv(root+'PREORE_FEM_Entrance_Exit_data.csv')
VAKI_Size_Weigth_initial = pd.read_csv(root+'PREORE_VAKI_Size_Weigth.csv')
VAKI_Size_Weigth_unique = VAKI_Size_Weigth_initial['observed_timestamp'].str[:10].unique()

# FISH_WEIGHT_unive = pd.read_csv(root+'UNIVE_MOD1.FISH_WEIGHT.unive_trout.3__56.csv')
# FISH_WEIGHT_unive_date_unique = FISH_WEIGHT_unive['observed_timestamp'].str[:10].unique()
# rainbow_trout_model = rainbow_trout_model()

dynamic_weight = dict()
time_row_index = 0
# 2019-11-14 09:58:57+00:00
df_timeWindow.at[1, 'start_date'] = '2019-11-15'

# We use Vaki weights
for _,time_row in df_timeWindow.iterrows(): 
    rainbow_trout = rainbow_trout_model(time_row.sampling_rate_per_day)
    df_data['Entrance_timestamp'] = pd.to_datetime(df_data['Entrance_timestamp'])
    VAKI_Size_Weigth_initial['observed_timestamp'] = pd.to_datetime(VAKI_Size_Weigth_initial['observed_timestamp'])
    
    
    start_date = pd.to_datetime(time_row.start_date)
    end_date = pd.to_datetime(time_row.end_date)
    
    
    df_data_temp = df_data[(df_data['Entrance_timestamp'].dt.date >= start_date.date()) & 
                           (df_data['Entrance_timestamp'].dt.date <= end_date.date())]
    
    VAKI_Size_Weigth = VAKI_Size_Weigth_initial[(VAKI_Size_Weigth_initial['observed_timestamp'].dt.date >= start_date.date()) &
                                                (VAKI_Size_Weigth_initial['observed_timestamp'].dt.date <= end_date.date())]
    
    # df_temperature_data = df_data_temp[['Entrance_timestamp', 'PREORE_FEM_ENTRANCE-Temp [Â°C]']].copy()
    df_temperature_data = df_data_temp.copy()
    df_temperature_data = df_temperature_data.reset_index(drop=True)
    VAKI_Size_Weigth = VAKI_Size_Weigth.reset_index(drop=True)
    
    # Plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(VAKI_Size_Weigth['observed_timestamp'], VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'], marker='.', linestyle='-')
    # plt.title('Fish Weight Over Time')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Weight (g)')
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # =========================================intial weights
    observed_weights = pd.Series(VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'])  # Replace with actual data
    mean_weight = observed_weights.mean()
    std_dev_weight = observed_weights.std() 
    # Step 2: Draw samples from the normal distribution
    num_samples = 1  # Specify the number of initial weights you want to generate
    initial_weights = np.random.normal(loc=mean_weight, scale=std_dev_weight, size=num_samples)
    mean_random_weights = math.ceil(sum(initial_weights)/num_samples)
    # =======================================================        
    initial_weight_data = VAKI_Size_Weigth.loc[0]   
    df_temperature_data = df_temperature_data[df_temperature_data['Entrance_timestamp'] >= initial_weight_data['observed_timestamp']].reset_index(drop=True)
    df_temperature_data['Fish_Weight']=None
         
    # fish_weight = mean_random_weights 
    fish_weight = initial_weight_data['PREORE_VAKI-Weight [g]']
    for index,row in df_temperature_data.iterrows():
        df_temperature_data.at[index, 'Fish_Weight'] = fish_weight  
        water_temperature = row['PREORE_FEM_ENTRANCE-Temp [Â°C]'] 
        Energy_Acquisition_A = rainbow_trout.Energy_Acquisition(fish_weight,water_temperature)   
        Catabolic_component_C = rainbow_trout.Catabolic_component(fish_weight,water_temperature)
        Somatic_tissue_energy_content_Epsilon = rainbow_trout.Total_energy_input(fish_weight)
        delta_t = 24/(rainbow_trout.constants_obj.delta_t*time_row.sampling_rate_per_day)
        Dynamic_individual_weight = rainbow_trout.Dynamic_individual_weight(Energy_Acquisition_A,Catabolic_component_C,Somatic_tissue_energy_content_Epsilon,delta_t)
        fish_weight += Dynamic_individual_weight
        
    print("================================================================")
    print("Time Window: " + str(time_row_index))
    print("Initial weight: "+ str(mean_random_weights))
    print('Mean of fish weights predicted by IB_PD model: '+ str(np.mean(df_temperature_data['Fish_Weight'])))
    print('Mean of fish weights measured by BD: '+ str(np.mean(VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'])))
    my_dict = {'start_date': start_date, 
               'end_date': end_date, 
               'sampling_rate_per_day': time_row.sampling_rate_per_day,
               'initial_weight': mean_random_weights,
               'df': df_temperature_data
               }
    dynamic_weight[time_row_index] = my_dict
    time_row_index += 1

    # comparison
    
    # Observed and predicted weights (example data)
    observed_weights = pd.Series(VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'])  # Observed fish weights
    predicted_weights = pd.Series(df_temperature_data['Fish_Weight'])  # Predicted fish weights
    
    # Calculate mean and standard deviation for observed and predicted weights
    observed_mean = observed_weights.mean()
    predicted_mean = predicted_weights.mean()
    
    observed_std = observed_weights.std()
    predicted_std = predicted_weights.std()
    
    # RMSE for the mean weights
    rmse_mean = np.sqrt(mean_squared_error([observed_mean], [predicted_mean]))

    # RMSE for the standard deviation
    rmse_std = np.sqrt(mean_squared_error([observed_std], [predicted_std]))
    
    # Display the results
    print("Goodness-of-Fit Indices:")
    print(f"RMSE (Mean Weights): {rmse_mean}")
    print(f"RMSE (Standard Deviation): {rmse_std}")

print('done')
with open(root + 'results/mechanistic_dynamic_individual_weight.pkl', 'wb') as file:
    pickle.dump(dynamic_weight, file)
    
    
    
    

