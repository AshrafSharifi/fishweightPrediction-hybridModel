import os
import pandas as pd


class constants:
    # Maximum ingestion rate : 1hour : 60 min
    delta_t :int = 1
    
    # Maximum ingestion rate
    I_max: float = 1.831  # kJ (g fish m d)^-1
    
    # Feeding catabolism coefficient
    alpha: float = 0.56  # - 
    
    # Assimilation coefficient for protein
    beta_p: float = 0.93  # - 
    
    # Assimilation coefficient for lipid
    beta_l: float = 0.94  # - 
    
    # Assimilation coefficient for carbohydrate
    beta_c: float = 0.67  # - 
    
    # Energy content of protein
    epsilon_p: float = 23.6  # kJ g protein^-1 
    
    # Energy content of lipid
    epsilon_l: float = 36.2  # kJ g lipid^-1 
    
    # Energy content of carbohydrate
    epsilon_c: float = 17.2  # kJ g carb^-1 
    
    # Energy consumed by the respiration of 1g of oxygen
    epsilon_O2: float = 13.6  # kJ g O2^-1 
    
    # Intercept of energy content of somatic tissue equation
    a_epsilon_T: float = 5.763  # kJ g fish^-1 
    
    # Angular coefficient of somatic energy tissue equation
    b_epsilon_T: float = 0.9852  # kJ g fish^-2 
    
    # Temperature coefficient for the fasting catabolism
    p_k: float = 0.06  # °C^-1 
    
    # Fasting catabolism at 0°C
    k_0: float = 7.2e-4  # g O2 (g fish n d)^-1 
    
    # Weight exponent for the anabolism
    m: float = 0.6  # - 
    
    # Shape coefficient of the function H(T_w)
    b: float = 0.2  # - 
    
    # Weight exponent for the catabolism
    n: float = 1.0  # - 
    
    # Maximum lethal temperature for O. mykiss
    T_m: float = 25.0  # °C 
    
    # Optimal temperature for O. mykiss
    T_o: float = 16.0  # °C 
    
    # Lowest feeding temperature for O. mykiss
    T_lf: float = 2.0  # °C 