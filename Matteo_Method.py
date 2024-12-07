import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pickle
from general import general
import rainbow_trout_model
import Custom_plots

def diff_equation_set2_(t, W, rainbow_trout, temperature_app, g_app):
    """
    Differential equations to model fish growth dynamics.

    Parameters:
        t (float): Current time step.
        W (float): Current weight of the fish.
        parameters (np.array): Array of model parameters.
        alfa (float): Feeding catabolism coefficient array.
        k0 (float): Fasting catabolism array.
        temperature_app (function): Interpolation function for temperature.
        g_app (function): Interpolation function for feeding rate.

    Returns:
        List of derivatives and additional metrics.
    """
    
    # Get current temperature and feeding rate
    Temperature = temperature_app(t)
    G = g_app(t)
    
    parameters=rainbow_trout.constants_obj
    epstiss = rainbow_trout.constants_obj.a_epsilon_T
    alpha = np.random.normal(0.579, 0.059, 1)[0]
    k0 = np.random.normal(parameters.k_0, 1.79e-06, 1)
     
    Taa = parameters.T_lf
    # Energy content of somatic tissue


    # Temperature-based functions
    fgT = (((parameters.T_m - Temperature) / (parameters.T_m - parameters.T_o)) ** (parameters.beta_c * (parameters.T_m - parameters.T_o))) * np.exp(parameters.beta_c * (Temperature - parameters.T_o))
    frT = np.exp(parameters.p_k * Temperature)

    # Energy content of the diet
    diet = (parameters.Pcont * parameters.epsilon_p * parameters.beta_p) + (parameters.Lcont * parameters.epsilon_l * parameters.beta_l) + (parameters.Ccont * parameters.epsilon_c * parameters.beta_c)
    
    # Stop feeding if temperature is too low
    if Temperature <= Taa:
        G = 0
    
    # Compute ingestion and growth
    if G == 0:
        ing = parameters.I_max * fgT * (W ** parameters.m)
        dW = ((ing * (1 - alpha)) - (parameters.epsilon_O2 * k0 * frT * (W ** parameters.n))) / epstiss
    else:
        ing = G * diet
        dW = ((ing * (1 - alpha)) - (parameters.epsilon_O2 * k0 * frT * (W ** parameters.n))) / epstiss

    # Anabolism and Catabolism
    anab = ing * (1 - alpha)
    catab = parameters.epsilon_O2 * k0 * frT * (W ** parameters.n)

    return [dW], {
        "Anab": anab,
        "Catab": catab,
        "I": ing / diet if diet != 0 else 0,
        "H_T": fgT,
        "k_T": frT,
        "Temp": Temperature,
        "Feed": G
    }


def diff_equation_set2(t, W, rainbow_trout, temperature_app, g_app):
    """
    Differential equations to model fish growth dynamics.

    Parameters:
        t (float): Current time step.
        W (float): Current weight of the fish.
        parameters (np.array): Array of model parameters.
        alfa (float): Feeding catabolism coefficient array.
        k0 (float): Fasting catabolism array.
        temperature_app (function): Interpolation function for temperature.
        g_app (function): Interpolation function for feeding rate.

    Returns:
        List of derivatives and additional metrics.
    """  
    # Get current temperature and feeding rate
    water_temperature = temperature_app(t)
    fish_weight = W
    Feeding_Amount = g_app(t)
    
    Energy_Acquisition_A = rainbow_trout.Energy_Acquisition(fish_weight,water_temperature)   
    Catabolic_component_C= rainbow_trout.Catabolic_component(fish_weight,water_temperature)
    Somatic_tissue_energy_content_Epsilon = rainbow_trout.Total_energy_input(fish_weight)
    # Feeding_Amount = rainbow_trout.Input_ration(fish_weight,water_temperature)
    # row = merged_df.loc[index]
    # delta_t = 24/(rainbow_trout.constants_obj.delta_t*sampling_rate_per_day)
    Dynamic_individual_weight = rainbow_trout.Dynamic_individual_weight(Energy_Acquisition_A,Catabolic_component_C,Somatic_tissue_energy_content_Epsilon,1)


    return Dynamic_individual_weight, {
        "Anab": Energy_Acquisition_A,
        "Catab": Catabolic_component_C,
        "Temp": water_temperature,
        "Feed": Feeding_Amount
    }
with open('data/Preore_Dataset/results/dynamic_individual_weight.pkl', 'rb') as file:
    data = pickle.load(file)

data['data_contextual_weight']['mathematical_weights'] = 0
for i in range(len(data)-1):
    df_data_temp = data[i]['data_contextual_weight']
    df_data_temp = df_data_temp[['PREORE_FEM_ENTRANCE-Temp [Â°C]','PREORE_VAKI-Weight [g]']]
        
    rainbow_trout = rainbow_trout_model(24)
    
    temperature_data = df_data_temp['PREORE_FEM_ENTRANCE-Temp [Â°C]'].to_numpy()
    actual_values = df_data_temp['PREORE_VAKI-Weight [g]'].to_numpy()
    time_steps = np.arange(len(temperature_data))  # Assume evenly spaced for simplicity
    temperature_app = interp1d(time_steps, temperature_data, kind='linear', fill_value="extrapolate")
    df_data_temp['Ration_Per_Hour'] = df_data_temp.apply(
        lambda row: rainbow_trout.Input_ration(row['PREORE_VAKI-Weight [g]'], row['PREORE_FEM_ENTRANCE-Temp [Â°C]']),
        axis=1
    )
    g_app = interp1d(time_steps, df_data_temp['Ration_Per_Hour'], kind='linear', fill_value="extrapolate")
    # =========================================intial weights
    mean_weight = actual_values.mean()
    std_dev_weight = actual_values.std() 
    # Step 2: Draw samples from the normal distribution
    num_samples = 1  # Specify the number of initial weights you want to generate
    W0 = np.random.normal(loc=mean_weight, scale=std_dev_weight, size=num_samples)[0]
    # =======================================================        
    
    # W0 = df_data_temp['PREORE_VAKI-Weight [g]'][0]
    t_span = (0, len(time_steps))
    t_eval = np.linspace(0, len(time_steps), len(time_steps))  # Matching time steps
    
    solution = solve_ivp(
        fun=lambda t, W: diff_equation_set2(t, W, rainbow_trout, temperature_app, g_app)[0],
        t_span=t_span,
        y0=[W0],
        t_eval=np.linspace(0, len(time_steps), len(time_steps))
    )
    # Extract solution
    weights = solution.y[0]
    time = solution.t
    mse2,mae2,mape2= general.compute_metrics(weights, actual_values)
    plots = Custom_plots(weights,actual_values)
    plots.plot_all()
    
    
    
