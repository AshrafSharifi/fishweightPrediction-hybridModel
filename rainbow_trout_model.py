import pandas as pd
from constants import constants
import math
import numpy as np
import matplotlib as plt
from scipy.interpolate import interp2d

class rainbow_trout_model:
    def __init__(self,sampling_per_day):
        self.constants_obj = constants(sampling_per_day)

    
    def Dynamic_individual_weight(self,A,C,Epsilon,delta_t):
        return (A-C)/self.constants_obj.a_epsilon_T
    # Equation(4) : calculates the energy available for growth after accounting for energy losses
    def Energy_Acquisition(self,w,Tw):
        I = self.Energy_Intake_BasedOn_Temperature_and_Weight(w, Tw)
        A= (1 - self.constants_obj.alpha)*I
        return A

    # Equation(5) : optimal amount of feed the fish should consume based on their weight and the water temperature
    def Optimal_Ingestion_Rate(self,w,Tw):
        I_opt= self.constants_obj.I_max * self.Temperature_function(Tw) * (w**self.constants_obj.m)
        return I_opt
        
    # Equation(6) : Ingestion rate based on the current water temperature, reflecting the physiological responses of the fish
    # def Temperature_function(self, Tw):
    #     part1 = self.constants_obj.T_m - Tw
    #     part2 = self.constants_obj.T_m - self.constants_obj.T_o
    #     part3 = self.constants_obj.b *(self.constants_obj.T_m - self.constants_obj.T_o)
    #     H_tw = (part1 / part2) ** part3
    #     part4 = self.constants_obj.b *(Tw - self.constants_obj.T_o)
    #     H_tw = H_tw * math.exp(part4)
    #     return H_tw
    def Temperature_function(self, Tw):
        """
        Calculate the temperature-dependent feeding response H(Tw).
        """
        T_m = self.constants_obj.T_m  # Maximum lethal temperature (°C)
        T_o = self.constants_obj.T_o  # Optimal temperature (°C)
        b = self.constants_obj.b      # Shape coefficient of H(Tw)
    
        # Edge case: Tw outside biological range
        if Tw < self.constants_obj.T_lf or Tw > T_m:
            return 0
    
        # Avoid division by zero for edge case where T_m == T_o
        if T_m == T_o:
            raise ValueError("T_m and T_o cannot be equal.")
    
        # Compute each term in the equation
        term1 = (T_m - Tw) / (T_m - T_o)
        term1 = max(term1, 1e-10)  # Prevent zero division or log(0)
        power_term = term1 ** (b * (T_m - T_o))
    
        exp_term = math.exp(b * (Tw - T_o))
    
        # Combine terms
        H_Tw = power_term * exp_term
        return H_Tw

    
    # Equation(7) : Energy Intake(I)
    def Energy_Intake_BasedOn_Temperature_and_Weight(self, wm, Tw):
        I_ration = self.Input_ration(wm,Tw)

        # Energy_ration = self.IMax_HTw(Tw,wm)
        I_opt = self.Optimal_Ingestion_Rate(wm, Tw)
        
        Energy_ration = self.Energy_Intake_From_Feed(I_ration)
        
        print(Energy_ration)
        print(I_opt)
        print("-----------------------------")
        if I_ration==0:
            return I_opt
        
        I = None
        if Tw < self.constants_obj.T_lf:
            I = 0
        elif Tw >= self.constants_obj.T_lf and I_opt >= Energy_ration:
            I = Energy_ration
        elif Tw >= self.constants_obj.T_lf and I_opt <= Energy_ration:
            I = I_opt
            
        if I == None:
            H=0
        return I
            
    # Equation(8) : Energy intake from feed ration        
    def Energy_Intake_From_Feed(self,I):
        return  I * self.Energy_Density_of_Feed()
    
    # Equation(8) : Energy density
    def Energy_Density_of_Feed(self):
        # Nutritional content
        # Pcont = [0.48, 0.46, 0.45]
        # Lcont = [0.24, 0.25, 0.26]
        # Ccont = [0.15, 0.155, 0.16]
        
        # # Energy computation
        # E_feed = [
        #     (self.constants_obj.epsilon_p*self.constants_obj.beta_p * P + self.constants_obj.epsilon_c*self.constants_obj.beta_c * C + self.constants_obj.epsilon_l*self.constants_obj.beta_l * L)
        #     for P, C, L in zip(Pcont, Ccont, Lcont)
        # ]
        E_feed = ((self.constants_obj.Pcont*self.constants_obj.epsilon_p*self.constants_obj.beta_p) + 
                  (self.constants_obj.Ccont*self.constants_obj.epsilon_c*self.constants_obj.beta_c) +
                  (self.constants_obj.Lcont*self.constants_obj.epsilon_l*self.constants_obj.beta_l))
        return E_feed
    
    # Equation 9 & Equation 10: Catabolic component of fish metabolism
    def Catabolic_component(self,w,Tw):
        C = self.constants_obj.epsilon_O2 * self.constants_obj.k_0 * math.exp(self.constants_obj.p_k*Tw) * (w**self.constants_obj.n)
        return C


    # Equation 11: Total energy input
    def Total_energy_input(self,w):
        # epsilon =  self.constants_obj.a_epsilon_T + self.constants_obj.b_epsilon_T * w
        epsilon =  self.constants_obj.a_epsilon_T * w
        return epsilon
    
    # def Input_ration(self,w,T):
    #     def get_value(df, weight, temperature):
    #         # Find the row for the weight range
    #         for i, weight_range in enumerate(df['Fish (g)']):
    #             min_weight, max_weight = map(float, weight_range.split('-'))
    #             if min_weight <= weight <= max_weight:
    #                 # Find the closest available temperature column
    #                 closest_temp = min(df.columns[3:], key=lambda x: abs(float(x) - temperature))
    #                 return df.at[i, closest_temp]
    #         return None  # If no match found
    #     df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
    #     value = get_value(df_feedingTable, w, T)
    #     if value==None:
    #         print('error')
    #     return value
    
    def IMax_HTw(self,T,w):
        # Constants
        eps_prot = self.constants_obj.epsilon_p #23600   [J/g_prot]
        eps_carb = self.constants_obj.epsilon_c #17200   [J/g_carb]
        eps_lipi = self.constants_obj.epsilon_l #36200   [J/g_lip]

        # Nutritional content
        Pcont = [0.48, 0.46, 0.45]
        Lcont = [0.24, 0.25, 0.26]
        Ccont = [0.15, 0.155, 0.16]

        # Energy computation
        energy = [
            (eps_prot * P*self.constants_obj.beta_p + eps_carb * C*self.constants_obj.beta_c + eps_lipi * L*self.constants_obj.beta_l)
            for P, C, L in zip(Pcont, Ccont, Lcont)
        ]

        
        # Size categories
        taglia_media = np.array([70, 150, 300, 500, 700, 900, 1100, 1300,1500,1700,1900])
        df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
        percentuali_mtx = np.array(df_feedingTable.iloc[:, 2:])
        # Percentage matrix
        # percentuali_mtx1 = np.array([
        #     [0.54, 0.63, 0.8, 0.93, 1.19, 1.46, 1.56, 1.62, 1.54],
        #     [0.47, 0.55, 0.7, 0.81, 1.04, 1.28, 1.37, 1.42, 1.35],
        #     [0.41, 0.49, 0.61, 0.71, 0.91, 1.13, 1.2, 1.25, 1.19],
        #     [0.36, 0.43, 0.54, 0.63, 0.80, 0.99, 1.06, 1.1, 1.05],
        #     [0.32, 0.38, 0.48, 0.55, 0.71, 0.87, 0.93, 0.97, 0.92],
        #     [0.28, 0.33, 0.42, 0.49, 0.62, 0.77, 0.82, 0.85, 0.81],
        # ])

        # Energy matrix
        energia_mtx = np.zeros_like(percentuali_mtx)
        energia_mtx[0] = percentuali_mtx[0] * energy[0] / 100
        energia_mtx[1:3] = percentuali_mtx[1:3] * energy[1] / 100
        energia_mtx[3:] = percentuali_mtx[3:] * energy[2] / 100

        # Adjusted energy based on fish size
        a_TW_mtx = np.zeros_like(energia_mtx)
        for i, size in enumerate(taglia_media):
            a_TW_mtx[i] = energia_mtx[i] / (size ** (-1 / 3))
        temperatures = np.array(df_feedingTable.columns[2:],dtype='int')  
        # Interpolator over taglia_media (weight) and temperature
        interpolator = interp2d(
            temperatures,         # x-axis: temperature
            taglia_media,  # y-axis: fish size (weights)
            a_TW_mtx,                  # z-values: matrix values
            kind='linear',             # Linear interpolation
        )
    
        # Interpolate and return the value
        return interpolator(T, w)[0]/ 24
    
    
    # def IMax_HTw(self,T,w):
        


    #     # Size categories
    #     taglia_media = np.array([70, 150, 300, 500, 700, 900, 1100, 1300,1500,1700,1900])
    #     df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
    #     percentuali_mtx = np.array(df_feedingTable.iloc[:, 2:])
        
    #     # Energy matrix
    #     energia_mtx = percentuali_mtx*10
        

    #     # Adjusted energy based on fish size
    #     a_TW_mtx = np.zeros_like(energia_mtx)
    #     for i, size in enumerate(taglia_media):
    #         a_TW_mtx[i] = energia_mtx[i] / (size ** (-1 / 3))
    #     temperatures = np.array(df_feedingTable.columns[2:],dtype='int')  
    #     # Interpolator over taglia_media (weight) and temperature
    #     interpolator = interp2d(
    #         temperatures,         # x-axis: temperature
    #         taglia_media,  # y-axis: fish size (weights)
    #         a_TW_mtx,                  # z-values: matrix values
    #         kind='linear',             # Linear interpolation
    #     )
    
    #     # Interpolate and return the value
    #     return interpolator(T, w)[0]


        
    
    # def Input_ration(self,w, T):
    #     def get_value(df, weight, temperature):
    #         # Parse weight ranges and create a list of (min, max) tuples
    #         weight_ranges = [
    #             tuple(map(float, weight_range.split('-'))) 
    #             for weight_range in df['Fish (g)']
    #         ]
    
    #         # Check if the weight falls in any range
    #         for i, (min_weight, max_weight) in enumerate(weight_ranges):
    #             if min_weight <= weight <= max_weight:
    #                 # Find the closest available temperature column
    #                 closest_temp = min(df.columns[3:], key=lambda x: abs(float(x) - temperature))
    #                 return df.at[i, closest_temp]
            
    #         # Handle out-of-range weights
    #         if weight < weight_ranges[0][0]:  # Below minimum weight range
    #             closest_index = 0
    #         elif weight > weight_ranges[-1][1]:  # Above maximum weight range
    #             closest_index = len(weight_ranges) - 1
    #         else:
    #             # If weight is not mapped (edge case), this shouldn't happen
    #             return None
            
    #         # Find the closest available temperature column
    #         closest_temp = min(df.columns[3:], key=lambda x: abs(float(x) - temperature))
    #         return df.at[closest_index, closest_temp]

    #     # Load feeding table0
    #     df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
    
    #     # Get value from the table, Kg feed per 100 kg fish per day
    #     value = get_value(df_feedingTable, w, T)
    #     grams_per_fish_per_day = (value/100)
    #     grams_per_fish_per_hour = (grams_per_fish_per_day / self.constants_obj.sampling_per_day)
    
    #     # Handle cases where value is None
    #     # if grams_per_fish_per_hour is None:
    #     #     print('Error: Could not find a suitable ration.')
    #     return grams_per_fish_per_day
    
    def Input_ration(self,w, T):
        def get_value(df, weight, temperature):
            # Parse weight ranges and create a list of (min, max) tuples
            weight_ranges = [
                tuple(map(float, weight_range.split('-'))) 
                for weight_range in df['Fish (g)']
            ]
    
            # Check if the weight falls in any range
            for i, (min_weight, max_weight) in enumerate(weight_ranges):
                if min_weight <= weight <= max_weight:
                    # Find the closest available temperature column
                    closest_temp = min(df.columns[3:], key=lambda x: abs(float(x) - temperature))
                    return df.at[i, closest_temp]
            
            # Handle out-of-range weights
            if weight < weight_ranges[0][0]:  # Below minimum weight range
                closest_index = 0
            elif weight > weight_ranges[-1][1]:  # Above maximum weight range
                closest_index = len(weight_ranges) - 1
            else:
                # If weight is not mapped (edge case), this shouldn't happen
                return None
            
            # Find the closest available temperature column
            closest_temp = min(df.columns[3:], key=lambda x: abs(float(x) - temperature))
            return df.at[closest_index, closest_temp]

        # Load feeding table0
        df_feedingTable = pd.read_csv('data/Preore_Dataset/feedingtable.csv')
    
        # Get value from the table, Kg feed per 100 kg fish per day
        I = get_value(df_feedingTable, w, T)
        if I==None:
            return 0
        I = I/24 #kg feed per 100kg fish per h
        I = (I*(w/1000))/100 #kg food for the given fish weight
        I = I*1000 #g food for the given fish weight
        return I
    
    # def Input_ration(self, T, w):
    #     # Load feeding data
    #     df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
    #     percentuali_mtx = np.array(df_feedingTable.iloc[:, 2:])
        
    #     # Convert percentages to grams of feed per gram of fish
    #     feed_per_gram_fish_hourly = (percentuali_mtx)  # Convert daily % to hourly g per g of fish
        
    #     # Size categories
    #     taglia_media = np.array([70, 150, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900])
    #     temperatures = np.array(df_feedingTable.columns[2:], dtype='int')
    
    #     # Interpolator over taglia_media (weight) and temperature
    #     interpolator = interp2d(
    #         temperatures,              # x-axis: temperature
    #         taglia_media,              # y-axis: fish size (weights)
    #         feed_per_gram_fish_hourly, # z-values: grams of feed per gram of fish
    #         kind='linear',             # Linear interpolation
    #     )
        
    #     I = interpolator(T, w)[0]  #kg feed per 100kg fish per d
    #     I = I/24 #kg feed per 100kg fish per h
    #     I = (I*(w/1000))/100 #kg food for the given fish weight
    #     I = I*1000 #g food for the given fish weight
        
    #     # Interpolate and return the value
    #     return I

        

            
            
        

