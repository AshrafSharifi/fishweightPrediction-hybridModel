import os
import pandas as pd


class constants:   
    def __init__(self, sampling_per_day=24):
        # Sampling frequency
        self.sampling_per_day = sampling_per_day
        
        # Maximum ingestion rate
        self.I_max = 76.29  # J (g fish m h)^-1
        
        # Fasting catabolism at 0°C
        self.k_0 = 	0.0000354  # g O2 (g fish n h)^-1
 
    delta_t :int = 1
    
    # protein fraction of food
    Pcont: float = 0.442  # - 
    
    # lipid fraction of food
    Lcont: float = 0.26  # - 
    
    # carbohydrate fraction of food
    Ccont: float = 0.158  # - 
    
    # Maximum ingestion rate
    # I_max: float = 1.831  # kJ (g fish m d)^-1  
    # I_max: float = 1.831/sampling_per_day  # kJ (g fish m h)^-1   
    
    # Feeding catabolism coefficient
    alpha: float = 0.579  # - 
    
    # Assimilation coefficient for protein
    beta_p: float = 0.93  # - 
    
    # Assimilation coefficient for lipid
    beta_l: float = 0.94  # - 
    
    # Assimilation coefficient for carbohydrate
    beta_c: float = 0.67  # - 
    
    # Energy content of protein
    # epsilon_p: float = 23.6  # kJ g protein^-1 
    epsilon_p: float = 23600  # J g protein^-1 
    
    # Energy content of lipid
    # epsilon_l: float = 36.2  # kJ g lipid^-1 
    epsilon_l: float = 36200  # J g lipid^-1 
    
    # Energy content of carbohydrate
    # epsilon_c: float = 17.2  # kJ g carb^-1 
    epsilon_c: float = 17200  # J g carb^-1 
    
    # Energy consumed by the respiration of 1g of oxygen
    # epsilon_O2: float = 13.6  # kJ g O2^-1 
    epsilon_O2: float = 13600  #  J g O2^-1 
    
    # Intercept of energy content of somatic tissue equation
    a_epsilon_T: float = 5763  # J g fish^-1 
    
    
    # Angular coefficient of somatic energy tissue equation
    b_epsilon_T: float = 9852  # J g fish^-2 
    
    # Temperature coefficient for the fasting catabolism
    p_k: float = 0.07  # °C^-1 
    
    # Fasting catabolism at 0°C
    # k_0: float = 7.2e-4  # g O2 (g fish n d)^-1 
    # k_0: float = 7.2e-4/24  # g O2 (g fish n h)^-1 
    
    # Weight exponent for the anabolism
    m: float = 0.67  # - 
    
    # Shape coefficient of the function H(T_w)
    b: float = 0.22  # - 
    
    # Weight exponent for the catabolism
    n: float = 0.8  # - 
    
    # Maximum lethal temperature for O. mykiss
    T_m: float = 25.0  # °C 
    
    # Optimal temperature for O. mykiss
    T_o: float = 15.0  # °C 
    
    # Lowest feeding temperature for O. mykiss
    T_lf: float = 2.0  # °C
    
    # Initial condition weight
    IC: float = 20	#g
    
    
    
 	
