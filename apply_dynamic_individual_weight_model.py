import pandas as pd
import numpy as np
from rainbow_trout_model import *
import json
import pickle

root = 'data/Preore_Dataset/'
df_timeWindow = pd.read_csv(root+'Time_window.csv')
with open(root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
    dynamic_individual_weight = pickle.load(file)

for i in range(0,len(dynamic_individual_weight)):
    weight_trend_item = dynamic_individual_weight[i]
    df = weight_trend_item['df']
    df['Entrance_timestamp'] = pd.to_datetime(df['Entrance_timestamp'])
    df['Date'] = df['Entrance_timestamp'].dt.date
    date_counts = df.groupby('Date').size()