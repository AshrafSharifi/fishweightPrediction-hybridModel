import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from general import general
from rainbow_trout_model import rainbow_trout_model
from Custom_plots import Custom_plots
import pandas as pd
import pickle




root = 'data/Preore_Dataset/'
results_path = root + 'results/'

df_timeWindow = pd.read_csv(root+'Time_window.csv')
df_data = pd.read_csv(root+'PREORE_FEM_Entrance_Exit_data.csv')
VAKI_Size_Weigth_initial = pd.read_csv(root+'PREORE_VAKI_Size_Weigth.csv')
VAKI_Size_Weigth_unique = VAKI_Size_Weigth_initial['observed_timestamp'].str[:10].unique()
df_Vaki_weights_daily_mean_initial = pd.read_csv(root+'PREORE_VAKI-Weight_dailymean.csv')
df_Vaki_weights_daily_mean_initial['date'] = pd.to_datetime(df_Vaki_weights_daily_mean_initial['observed_timestamp']).dt.date

# FISH_WEIGHT_unive = pd.read_csv(root+'UNIVE_MOD1.FISH_WEIGHT.unive_trout.3__56.csv')
# FISH_WEIGHT_unive_date_unique = FISH_WEIGHT_unive['observed_timestamp'].str[:10].unique()
# rainbow_trout_model = rainbow_trout_model()

df_combined = pd.DataFrame()
dynamic_weight = dict()
time_row_index = 0
# 2019-11-14 09:58:57+00:00
df_timeWindow.at[0, 'start_date'] = df_Vaki_weights_daily_mean_initial['date'].iloc[0]

df_timeWindow.at[1, 'start_date'] = '2019-11-15'
data = dict()
# We use Vaki weights
for _,time_row in df_timeWindow.iterrows(): 
    rainbow_trout = rainbow_trout_model(time_row.sampling_rate_per_day)
    start_date = pd.to_datetime(time_row.start_date)
    end_date = pd.to_datetime(time_row.end_date)
    df_Vaki_weights_daily_mean = df_Vaki_weights_daily_mean_initial[(df_Vaki_weights_daily_mean_initial['date'] >= start_date.date()) & 
                                                                    (df_Vaki_weights_daily_mean_initial['date'] <= end_date.date())].reset_index(drop=True)
    
    
    
    
    df_data['Entrance_timestamp'] = pd.to_datetime(df_data['Entrance_timestamp'])
    VAKI_Size_Weigth_initial['observed_timestamp'] = pd.to_datetime(VAKI_Size_Weigth_initial['observed_timestamp'])
    
    df_data_temp = df_data[(df_data['Entrance_timestamp'].dt.date >= start_date.date()) & 
                           (df_data['Entrance_timestamp'].dt.date <= end_date.date())].reset_index(drop=True)
    
    VAKI_Size_Weigth = VAKI_Size_Weigth_initial[(VAKI_Size_Weigth_initial['observed_timestamp'].dt.date >= start_date.date()) &
                                                (VAKI_Size_Weigth_initial['observed_timestamp'].dt.date <= end_date.date())].reset_index(drop=True)
    

    df_temperature_data = df_data_temp.copy()
    df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'] = general.interpolate_outliers(df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'],'Weight_dailymean [g]')
    VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'] = general.interpolate_outliers(VAKI_Size_Weigth['PREORE_VAKI-Weight [g]'],'Weight [g]')
    df_temperature_data['PREORE_FEM_ENTRANCE-Temp [Â°C]'] = general.interpolate_outliers(df_temperature_data['PREORE_FEM_ENTRANCE-Temp [Â°C]'],'Temp [Â°C]')
    
    df_temperature_data = df_temperature_data.sort_values(by='Entrance_timestamp')
    VAKI_Size_Weigth = VAKI_Size_Weigth.sort_values(by='observed_timestamp')
    
    merged_df = pd.merge_asof(
        df_temperature_data,
        VAKI_Size_Weigth,
        left_on='Entrance_timestamp',
        right_on='observed_timestamp',
        tolerance=pd.Timedelta('70min'),
        direction='nearest'
    ).dropna()
    
    
    temperature_data = merged_df['PREORE_FEM_ENTRANCE-Temp [Â°C]'].to_numpy()
    actual_values = merged_df['PREORE_VAKI-Weight [g]'].to_numpy()
    time_steps = np.arange(len(merged_df))  # Assume evenly spaced for simplicity
    
    
    # =========================================intial weights
    temp = df_Vaki_weights_daily_mean[df_Vaki_weights_daily_mean['date'] == start_date.date()]
    W0 = temp['PREORE_VAKI-Weight_dailymean [g]'].iloc[0]
    
    # mean_weight = actual_values.mean()
    # std_dev_weight = actual_values.std() 
    # # Step 2: Draw samples from the normal distribution
    # num_samples = 1  # Specify the number of initial weights you want to generate
    # W0 = np.random.normal(loc=mean_weight, scale=std_dev_weight, size=num_samples)[0]
    # W0 = actual_values[0]
    # =======================================================        
    t_span = (0, len(time_steps))
    t_eval = np.linspace(0, len(time_steps), len(time_steps))  # Matching time steps
    
    temperature_app = interp1d(time_steps, temperature_data, kind='linear', fill_value="extrapolate")
    merged_df['I_Ration_Per_SamplingFrequency'] = merged_df.apply(lambda row: rainbow_trout.Input_ration(row['PREORE_VAKI-Weight [g]'], row['PREORE_FEM_ENTRANCE-Temp [Â°C]']),axis=1)
    g_app = interp1d(time_steps, merged_df['I_Ration_Per_SamplingFrequency'], kind='linear', fill_value="extrapolate")
    
    solution = solve_ivp(
        fun=lambda t, W: rainbow_trout.diff_equation_set(t, W, rainbow_trout, temperature_app, g_app)[0],
        t_span=t_span,
        y0=[W0],
        t_eval=np.linspace(0, len(time_steps), len(time_steps))
    )
    
    Energy_Acquisition = []
    Catabolic_component = []
    Somatic_tissue_energy_content = []
    Feed_ration = []
    for t, W in zip(solution.t, solution.y[0]):
        _, metrics = rainbow_trout.diff_equation_set(t, W, rainbow_trout, temperature_app, g_app)
        Energy_Acquisition.append(metrics['Anab'])
        Catabolic_component.append(metrics['Catab'])
        Somatic_tissue_energy_content.append(metrics['Somatic_tissue_energy_content_Epsilon'])
        Feed_ration.append(metrics['I_ration'])
        
    merged_df['Energy_Acquisition(A)']=Energy_Acquisition
    merged_df['Catabolic_component(C)']=Catabolic_component
    merged_df['Somatic_tissue_energy_content(Epsilon)']=Somatic_tissue_energy_content
    merged_df['mathematical_computed_weight']=solution.y[0]
    merged_df['Feed_ration']=Feed_ration
    
    # Extract solution
    weights = solution.y[0]
    time = solution.t
    mse2,mae2,mape2= general.compute_metrics(weights, actual_values)
    plots = Custom_plots(weights,actual_values)
    plots.plot_all()
    
    
    df_combined = pd.concat([df_combined, merged_df], ignore_index=True)
    data[time_row_index] = {'start_date': start_date, 
               'end_date': end_date, 
               'sampling_rate_per_day': time_row.sampling_rate_per_day,
               'initial_weight': W0,
               'df':  merged_df.reset_index(drop=True)
               }
    
    
    time_row_index +=1
    
data['data_contextual_weight'] = df_combined.reset_index(drop=True)    
with open(results_path + 'dynamic_individual_weight.pkl', 'wb') as file:
    pickle.dump(data, file)
df_timeWindow.to_csv(root+'Time_window.csv')
    
    