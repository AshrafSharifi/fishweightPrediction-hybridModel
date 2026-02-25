import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_shap_comparison(run1_path, run2_path, label1="Run A", label2="Run B"):

    # 1. Load the CSV data
    df1 = pd.read_csv(run1_path)
    df2 = pd.read_csv(run2_path)
    
    # 2. Rename columns for merging
    df1 = df1.rename(columns={'Mean_Abs_SHAP': 'Importance_Run1'})
    df2 = df2.rename(columns={'Mean_Abs_SHAP': 'Importance_Run2'})
    
    # 3. Merge dataframes on the Feature column to ensure alignment
    comparison_df = pd.merge(df1, df2, on='Feature', how='outer').fillna(0)
    
    # Sort based on Run1
    comparison_df = comparison_df.sort_values(by='Importance_Run1', ascending=True)
    
    # 4. Feature renaming
    features = comparison_df['Feature'].values

    rename_map = {
        'PREORE_FEM_ENTRANCE-NO3 -N [mg/L]': 'Nitrate (NO₃-N)',
        'PREORE_FEM_ENTRANCE-NH4+ [mg/L]': 'Ammonium (NH₄⁺)',
        'PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]': 'Conductivity',
        'PREORE_FEM_ENTRANCE-ODO [mg/L]': 'Dissolved Oxygen',
        'Somatic_tissue_energy_content(Epsilon)': 'Somatic Tissue Energy (ε)',
        'PREORE_FEM_ENTRANCE-Sal [psu]': 'Salinity',
        'Feed_ration': 'Feed Ration',
        'PREORE_FEM_ENTRANCE-pH': 'pH',
        'Feed_ration_3d': 'Feed Ration (3‑day Avg)',
        'Catabolic_component(C)': 'Catabolic Component (C)',
        'Energy_Acquisition(A)': 'Energy Acquisition (A)',
        'PREORE_FEM_ENTRANCE-Temp [Â°C]': 'Temperature'
    }

    features = np.array([rename_map.get(f, f) for f in features])

    # 5. Plotting
    y = np.arange(len(features))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 10))

    # Bars
    rects1 = ax.barh(y - height/2, comparison_df['Importance_Run1'], height, label=label1)
    rects2 = ax.barh(y + height/2, comparison_df['Importance_Run2'], height, label=label2)

    # Axis labels and title (BOLD + BIG)
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=18, fontweight='bold')
    ax.set_title('Feature Importance Comparison', fontsize=20, fontweight='bold')

    # Y-axis ticks (BOLD + BIG)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=14, fontweight='bold')

    # X-axis tick labels (BOLD + BIG)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Legend (BIGGER + BOLD)
    ax.legend(fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig("feature_comparison_plot.png", dpi=300)
    plt.show()


# Example Usage:
path_campaign2 = "data/global_feature_importance.csv"
path_campaign3 = "data/global_feature_importance_modelfree_20260214_152727.csv"
plot_shap_comparison(path_campaign2, path_campaign3,
                     "Proposed Hybrid Data/Model Driven", "Pure Data-driven")
