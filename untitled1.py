import pandas as pd
from scipy import stats
import numpy as np

# Sample data (based on your dataframe)
data = {
    "Unnamed: 0": [0, 1, 2, 3, 4],
    "Fish (g)": ["40-100", "100-200", "200-400", "400-600", "600-800"],
    "MM": [3.0, 4.5, 4.5, 6.0, 6.0],
    2: [0.55, 0.49, 0.43, 0.37, 0.33],
    4: [0.65, 0.57, 0.50, 0.44, 0.39],
    6: [0.75, 0.65, 0.55, 0.47, 0.42],
    8: [0.95, 0.84, 0.74, 0.65, 0.57],
    10: [1.22, 1.07, 0.94, 0.83, 0.73],
    12: [1.50, 1.32, 1.16, 1.02, 0.90],
    14: [1.60, 1.41, 1.24, 1.09, 0.96],
    16: [1.67, 1.47, 1.29, 1.14, 1.00],
    18: [1.58, 1.40, 1.23, 1.08, 0.95],
}

df_feedingTable = pd.DataFrame(data)

# Step 1: Convert fish weight ranges into numeric midpoints
df_feedingTable['Weight_midpoint'] = df_feedingTable['Fish (g)'].apply(
    lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
)

# Step 2: Specify the weight ranges you're interested in (e.g., 1000, 1200, etc.)
target_weights = [800, 1000]

# Step 3: Interpolate the data for each temperature column
food_required = {}
list_o=[]
for temp in df_feedingTable.columns[3:]:  # Skip first few columns

    slope, intercept, _, _, _ = stats.linregress(df_feedingTable['Weight_midpoint'].values,  df_feedingTable[temp].values)
    
    for current, next_value in zip(target_weights, target_weights[1:]): 
        extended_weights = range(current, next_value, 100)
        extrapolated_rates = [slope * weight + intercept for weight in extended_weights]
        list_o.append(np.mean(extrapolated_rates)) 
    
   
# Step 4: Create a DataFrame for the interpolated results
interpolated_food_df = pd.DataFrame(food_required, index=target_weights)
interpolated_food_df.index.name = 'Weight (g)'

# Display the interpolated food requirements
print(interpolated_food_df)