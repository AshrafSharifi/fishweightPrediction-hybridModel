import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

root = 'data/Preore_Dataset/'
# Sample data (based on your dataframe)
data = {
    "Unnamed: 0": [0, 1, 2, 3, 4, 5],
    "Fish (g)": ["40-100", "100-200", "200-400", "400-600", "600-800", "800-1000"],
    "MM": [3.0, 4.5, 4.5, 6.0, 6.0, 6.0],
    2: [0.55, 0.49, 0.43, 0.37, 0.33, 0.29],
    4: [0.65, 0.57, 0.50, 0.44, 0.39, 0.34],
    6: [0.75, 0.65, 0.55, 0.47, 0.42, 0.37],
    8: [0.95, 0.84, 0.74, 0.65, 0.57, 0.50],
    10: [1.22, 1.07, 0.94, 0.83, 0.73, 0.64],
    12: [1.50, 1.32, 1.16, 1.02, 0.90, 0.79],
    14: [1.60, 1.41, 1.24, 1.09, 0.96, 0.84],
    16: [1.67, 1.47, 1.29, 1.14, 1.00, 0.88],
    18: [1.58, 1.40, 1.23, 1.08, 0.95, 0.84],
}

df_feedingTable = pd.DataFrame(data)
df_feedingTable = df_feedingTable.drop(['MM'],axis=1)
for index, row in df_feedingTable.iterrows():
    # Calculate the average decrement per step from the last two columns (16 and 18)
    step_decrease = (row[16] - row[18]) / 2
    df_feedingTable.at[index, 20] = row[18] - step_decrease  # Value for column 20
    df_feedingTable.at[index, 22] = df_feedingTable.at[index, 20] - step_decrease  # Value for column 22

# Step 1: Convert fish weight ranges into numeric midpoints
Weight_midpoint = df_feedingTable['Fish (g)'].apply(
    lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
)

# Step 2: Specify the weight ranges you're interested in (e.g., 1000, 1200, etc.)
target_weights = [1000, 1200, 1400, 1600, 1800, 2000]

# Step 3: Interpolate the data for each temperature column
food_required = {}

for current, next_value in zip(target_weights, target_weights[1:]): 
    
    new_row = [str(current)+'-'+str(next_value)]
    for temp in df_feedingTable.columns[2:]:  # Skip first few columns

        slope, intercept, _, _, _ = stats.linregress(Weight_midpoint.values,  df_feedingTable[temp].values)
    
    
        extended_weights = range(current, next_value, 100)
        extrapolated_rates = [slope * weight + intercept for weight in extended_weights]
        
        new_row.append(round(np.mean(extrapolated_rates), 2))
    
    Weight_midpoint[len(Weight_midpoint)]=extended_weights[1]
    new_row.insert(0,len(df_feedingTable))
    df_feedingTable.loc[len(df_feedingTable)] = new_row


fish_weights = df_feedingTable['Fish (g)']
temperature_columns = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22']
plt.figure(figsize=(14, 8))
for temp in temperature_columns:
    plt.plot(fish_weights, df_feedingTable[int(temp)], marker='o', label=f"{temp}°C")

# Adding labels and title
plt.xlabel('Fish Weight Range (g)')
plt.ylabel('Food Amount (g)')
plt.title('Feeding Rates at Different Temperatures')
plt.legend(title='Temperature (°C)')
plt.grid(True)
plt.show()

df_feedingTable.to_csv(root + "Final_feeding_table.csv", index=False)