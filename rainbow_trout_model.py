import os
import pandas as pd
from constants import constants
import math
import numpy as np


class rainbow_trout_model:
    def __init__(self):
        self.constants_obj = constants()
    
    def Dynamic_individual_weight(self,A,C,Epsilon,delta_t):
        return (((A-C)/Epsilon)*delta_t)
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
    def Temperature_function(self, Tw):
        part1 = self.constants_obj.T_m - Tw
        part2 = self.constants_obj.T_m - self.constants_obj.T_o
        part3 = self.constants_obj.b *(self.constants_obj.T_m - self.constants_obj.T_o)
        H_tw = (part1 / part2) ** part3
        part4 = self.constants_obj.b *(Tw - self.constants_obj.T_o)
        H_tw = H_tw * math.exp(part4)
        return H_tw
    
    # Equation(7) : Energy Intake(I)
    def Energy_Intake_BasedOn_Temperature_and_Weight(self, wm, Tw):
        I_ration = self.Input_ration(wm,Tw)
        I_opt = self.Optimal_Ingestion_Rate(wm, Tw)
        # if I_ration == None:
        #     return I_opt
        Energy_ration = self.Energy_Intake_From_Feed(I_ration)
        
        if Tw < self.constants_obj.T_lf:
            I = 0
        elif Tw >= self.constants_obj.T_lf and I_opt >= Energy_ration:
            I = Energy_ration
        elif Tw >= self.constants_obj.T_lf and I_opt <= Energy_ration:
            I = I_opt
        return I
            
    # Equation(8) : Energy intake from feed ration        
    def Energy_Intake_From_Feed(self,I):
        return  I * self.Energy_Density_of_Feed()
    
    # Equation(8) : Energy density
    def Energy_Density_of_Feed(self):
        E_feed = ((self.constants_obj.epsilon_p*self.constants_obj.beta_p) + 
                 (self.constants_obj.epsilon_c*self.constants_obj.beta_c) +
                 (self.constants_obj.epsilon_l*self.constants_obj.beta_l))
        return E_feed
    
    # Equation 9 & Equation 10: Catabolic component of fish metabolism
    def Catabolic_component(self,w,Tw):
        C = self.constants_obj.epsilon_O2 * self.constants_obj.k_0 * math.exp(self.constants_obj.p_k*Tw) * (w**self.constants_obj.n)
        return C


    # Equation 11: Total energy input
    def Total_energy_input(self,w):
        epsilon =  self.constants_obj.a_epsilon_T + self.constants_obj.b_epsilon_T * w
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

        # Load feeding table
        df_feedingTable = pd.read_csv('data/Preore_Dataset/Final_feeding_table.csv')
    
        # Get value from the table
        value = get_value(df_feedingTable, w, T)
    
        # Handle cases where value is None
        if value is None:
            print('Error: Could not find a suitable ration.')
        return value
        

            
            
        

