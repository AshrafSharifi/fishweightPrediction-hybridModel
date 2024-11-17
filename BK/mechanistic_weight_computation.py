import pandas as pd
import numpy as np
from rainbow_trout_model import *
import json
import pickle
root = 'data/Preore_Dataset/'
df_timeWindow = pd.read_csv(root+'Time_window.csv')
df_data = pd.read_csv(root+'PREORE_FEM_Entrance_Exit_data.csv')
VAKI_Size_Weigth_initial = pd.read_csv(root+'PREORE_VAKI_Size_Weigth.csv')
VAKI_Size_Weigth_unique = VAKI_Size_Weigth_initial['observed_timestamp'].str[:10].unique()

FISH_WEIGHT_unive = pd.read_csv(root+'UNIVE_MOD1.FISH_WEIGHT.unive_trout.3__56.csv')
FISH_WEIGHT_unive_date_unique = FISH_WEIGHT_unive['observed_timestamp'].str[:10].unique()
rainbow_trout_model = rainbow_trout_model()

dynamic_weight = dict()
time_row_index = 0
# We use Vaki weights
for _,time_row in df_timeWindow.iterrows():
    start_date = time_row.start_date
    end_date = time_row.end_date
    df_data_temp = df_data[df_data['Entrance_timestamp'].str[:10].between(start_date,end_date)]
    
    VAKI_Size_Weigth = VAKI_Size_Weigth_initial[VAKI_Size_Weigth_initial['observed_timestamp'].str[:10].between(start_date,end_date)]
    
    df_temperature_data = pd.DataFrame(df_data_temp[['Entrance_timestamp', 'PREORE_FEM_ENTRANCE-Temp [Â°C]']])

    df_temperature_data['Entrance_timestamp'] = pd.to_datetime(df_temperature_data['Entrance_timestamp'])
    VAKI_Size_Weigth['observed_timestamp'] = pd.to_datetime(VAKI_Size_Weigth['observed_timestamp'])
    

    df_temperature_data = df_temperature_data.sort_values('Entrance_timestamp')
    VAKI_Size_Weigth = VAKI_Size_Weigth.sort_values('observed_timestamp')
    
    # Merge with a tolerance of 70min 
    merged_df = pd.merge_asof(df_temperature_data, VAKI_Size_Weigth, 
                               left_on='Entrance_timestamp', 
                               right_on='observed_timestamp', 
                               direction='nearest', 
                               tolerance=pd.Timedelta('70min'))
    merged_df = merged_df.dropna() 
    merged_df['Energy_Acquisition(A)'] = merged_df.apply(lambda row: rainbow_trout_model.Energy_Acquisition(row['PREORE_VAKI-Weight [g]'], row['PREORE_FEM_ENTRANCE-Temp [Â°C]']),axis=1)
    
    merged_df['Catabolic_component(C)'] = merged_df.apply(lambda row: rainbow_trout_model.Catabolic_component(row['PREORE_VAKI-Weight [g]'], row['PREORE_FEM_ENTRANCE-Temp [Â°C]']),axis=1)
    
    merged_df['Somatic_tissue_energy_content(Epsilon)'] = merged_df.apply(lambda row: rainbow_trout_model.Total_energy_input(row['PREORE_VAKI-Weight [g]']),axis=1)
    
    merged_df['Dynamic_individual_weight'] = merged_df.apply(lambda row: rainbow_trout_model.Dynamic_individual_weight(row['Energy_Acquisition(A)'],row['Catabolic_component(C)'],row['Somatic_tissue_energy_content(Epsilon)']),axis=1)
    
    
    my_dict = {'start_date': start_date, 
               'end_date': end_date, 
               'sampling_rate_per_day': time_row.sampling_rate_per_day,
               'df': merged_df
               }
    dynamic_weight[time_row_index] = my_dict
    time_row_index += 1
    print(time_row_index)

# with open(root + 'results/dynamic_individual_weight.pkl', 'wb') as file:
#     pickle.dump(dynamic_weight, file)
    
    
    
    

