import pandas as pd
import numpy as np
from rainbow_trout_model import *
import json
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

root = 'data/Preore_Dataset/'
with open(root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
    data = pickle.load(file)
VAKI_Size_Weigth_initial = pd.read_csv(root+'PREORE_VAKI_Size_Weigth.csv')
VAKI_Size_Weigth_initial['observed_timestamp'] = pd.to_datetime(VAKI_Size_Weigth_initial['observed_timestamp'])    

df_combined = pd.DataFrame()

for i in range(0,len(data)):
    df_temperature_data = data[i]['df']
    start_date = data[i]['start_date']
    end_date = data[i]['end_date']
    VAKI_Size_Weigth = VAKI_Size_Weigth_initial[(VAKI_Size_Weigth_initial['observed_timestamp'].dt.date >= start_date.date()) & 
                                                (VAKI_Size_Weigth_initial['observed_timestamp'].dt.date <= end_date.date())]
    
    merged_df = pd.merge_asof(df_temperature_data, VAKI_Size_Weigth, 
                           left_on='Entrance_timestamp', 
                           right_on='observed_timestamp', 
                           direction='nearest', 
                           tolerance=pd.Timedelta('70min'))
    merged_df = merged_df.dropna() 
    data[i]['data_contextual_weight'] = merged_df.reset_index(drop=True)
    df_combined = pd.concat([df_combined, merged_df], ignore_index=True)
    
    

df_combined = df_combined.reset_index(drop=True)
data['data_contextual_weight'] = df_combined
with open(root + 'results/dynamic_individual_weight.pkl', 'wb') as file:
    pickle.dump(data, file)


