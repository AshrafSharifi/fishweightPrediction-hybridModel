import pandas as pd
from rainbow_trout_model import *
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score


root = 'data/Preore_Dataset/'
results_path = root + 'results/'

# Ensure results directory exists
os.makedirs(results_path, exist_ok=True)

# Load existing data
with open(results_path + 'mechanistic_dynamic_individual_weight.pkl', 'rb') as file:
    data = pickle.load(file)

# Load and preprocess VAKI Size and Weight data
VAKI_Size_Weigth_initial = pd.read_csv(root + 'PREORE_VAKI_Size_Weigth.csv')
VAKI_Size_Weigth_initial['observed_timestamp'] = pd.to_datetime(VAKI_Size_Weigth_initial['observed_timestamp'])
VAKI_Size_Weigth_initial = VAKI_Size_Weigth_initial.sort_values(by='observed_timestamp')

# Initialize an empty DataFrame to store combined data
df_combined = pd.DataFrame()
rainbow_trout_model = rainbow_trout_model()
# Loop through each time window in the data
for i in range(len(data)):
    df_temperature_data = data[i]['df']
    start_date = data[i]['start_date']
    end_date = data[i]['end_date']
    sampling_rate_per_day = data[i]['sampling_rate_per_day']

    # Filter VAKI data for the current time window
    VAKI_Size_Weigth = VAKI_Size_Weigth_initial[
        (VAKI_Size_Weigth_initial['observed_timestamp'].dt.date >= start_date.date()) &
        (VAKI_Size_Weigth_initial['observed_timestamp'].dt.date <= end_date.date())
    ]

    # Ensure df_temperature_data is sorted by timestamp for merge_asof
    df_temperature_data = df_temperature_data.sort_values(by='Entrance_timestamp')

    # Perform nearest merge within a 70-minute tolerance
    merged_df = pd.merge_asof(
        df_temperature_data,
        VAKI_Size_Weigth,
        left_on='Entrance_timestamp',
        right_on='observed_timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('70min')
    ).dropna()
    
    merged_df['Energy_Acquisition(A)']=None
    merged_df['Catabolic_component(C)']=None
    merged_df['Somatic_tissue_energy_content(Epsilon)']=None
    merged_df['Dynamic_individual_weight']=None
    
    for index,row in merged_df.iterrows():
        fish_weight = merged_df.at[index, 'PREORE_VAKI-Weight [g]']
        water_temperature = row['PREORE_FEM_ENTRANCE-Temp [Â°C]']
        merged_df.at[index,'Energy_Acquisition(A)'] = rainbow_trout_model.Energy_Acquisition(fish_weight,water_temperature)   
        merged_df.at[index,'Catabolic_component(C)'] = rainbow_trout_model.Catabolic_component(fish_weight,water_temperature)
        merged_df.at[index,'Somatic_tissue_energy_content(Epsilon)'] = rainbow_trout_model.Total_energy_input(fish_weight)
        merged_df.at[index,'Feeding_Amount'] = rainbow_trout_model.Input_ration(fish_weight,water_temperature)
        row = merged_df.loc[index]
        delta_t = 24/(rainbow_trout_model.constants_obj.delta_t*sampling_rate_per_day)
        merged_df.at[index,'Dynamic_individual_weight'] = rainbow_trout_model.Dynamic_individual_weight(row['Energy_Acquisition(A)'],row['Catabolic_component(C)'],row['Somatic_tissue_energy_content(Epsilon)'],delta_t)
        
    # Add merged data to the original dictionary entry and combined DataFrame
    data[i]['data_contextual_weight'] = merged_df.reset_index(drop=True)
    df_combined = pd.concat([df_combined, merged_df], ignore_index=True)

# Save the combined data back to the original data dictionary
data['data_contextual_weight'] = df_combined.reset_index(drop=True)

# Save updated data dictionary to pickle file
with open(results_path + 'dynamic_individual_weight.pkl', 'wb') as file:
    pickle.dump(data, file)
