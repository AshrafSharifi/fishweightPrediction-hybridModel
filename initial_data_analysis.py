import pandas as pd
import numpy as np
import pyreadr as pr
import os
import matplotlib.pyplot as plt

# FishFarmingData = pd.read_csv('data/FishFarmingData.csv')
# plt.figure(figsize=(10, 6))
# plt.plot(FishFarmingData['Taglia (g)'], label='Biomass (kg)', color='green')
# plt.xlabel('Index or Time')  # Label based on the x-axis column (e.g., time if available)
# plt.ylabel('Biomass (kg)')
# plt.title('Biomass Over Time')
# plt.legend()
# plt.show()
# print(FishFarmingData.head())
# print(FishFarmingData.columns)

# FishFarmingDataWithCampaign = pd.read_csv('data/FishFarmingDataWithCampaign.csv')
# print(FishFarmingDataWithCampaign.head())
# print(FishFarmingDataWithCampaign.columns)

FishFarmingDataWithTimestamp = pd.read_csv('data/FishFarmingDataWithTimestamp.csv')
# print(FishFarmingDataWithTimestamp.head())
# print(FishFarmingDataWithTimestamp.columns)



dirs = os.listdir('data/R')
# print(dirs)
df_FeedData = pd.DataFrame()
df_FishCounts = pd.DataFrame()
for dir_name in dirs:
    dirs = os.listdir('data/R/'+dir_name+'/')[0]
    path = 'data/R/'+dir_name+'/'+dirs
    dirs = os.listdir(path)
    Feed_file = [item for item in dirs if 'FEED' in item][0]
    Feed_Data = pr.read_r(path+'/'+Feed_file)[None] 
    df_FeedData = pd.concat([df_FeedData, Feed_Data], ignore_index=True)
    No_file = [item for item in dirs if 'Num' in item]
    if len(No_file)!=0:
        Fishcount_Data = pd.read_csv(path+'/'+No_file[0]) 
        df_FishCounts = pd.concat([df_FishCounts, Fishcount_Data], ignore_index=True)
    print(dirs)
    

