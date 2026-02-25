from dataclasses import dataclass
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from general import general
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from torch.utils.tensorboard import SummaryWriter
from scipy.signal import savgol_filter
import tyro
import matplotlib.pyplot as plt
from ModelClass import ModelClass
from Custom_plots import Custom_plots
from general import general


# ============================================================
# ARGS
# ============================================================

@dataclass

@dataclass
class Args:
    # Model
    prediction_Method: str = "LSTM_CNN"

    # Training
    epochs: int = 80
    batch_size: int = 16
    patience: int = 8
    dropout: float = 0.2
    learning_rate: float = 0.001
    history_hours = 72
    sampling_minutes = 20
    # Window length (hours)
    timesteps: int = 72 *3    # 72-hour history

    # Cross-validation
    n_splits: int = 3      # IMPORTANT: only 3 folds for 30 days
    verbose: int = 0
    num_splits: int = 1     # For incremental learning (optional)
    # Preprocessing
    scale_flag: bool = True

    # Paths
    root: str = "data/Preore_Dataset/"
    path: str = ""
    model_file: str = ""
    run_name: str = ""
    selected_campagin: int = 2

    # Internals
    feature_names: list[str] | None = None
    scaler_x: MinMaxScaler = MinMaxScaler()
    reducedFeature: bool = True
    
  

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = []

    



def clean_daily_weights(df, weight_col='PREORE_VAKI-Weight [g]', date_col='Entrance_timestamp'):
    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Extract date
    df['date_only'] = df[date_col].dt.date

    daily_values = []

    for date, group in df.groupby('date_only'):
        w = group[weight_col].values

        # Remove extreme outliers (1st–99th percentile)
        low, high = np.percentile(w, [1, 99])
        w = np.clip(w, low, high)

        # Use median as daily representative weight
        daily_weight = np.median(w)

        daily_values.append({
            "date": date,
            "daily_weight": daily_weight
        })

    # Build daily dataframe
    df_daily = pd.DataFrame(daily_values)
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    # Smooth daily weight (7-day window)
    if len(df_daily) >= 7:
        df_daily["smooth_weight"] = savgol_filter(
            df_daily["daily_weight"], 
            window_length=7, 
            polyorder=2
        )
    else:
        df_daily["smooth_weight"] = df_daily["daily_weight"]

    return df_daily["smooth_weight"].values


# ============================================================
# DAILY AGGREGATION + LOG-GROWTH TARGET
# ============================================================

def prepare_hourly_and_daily_data(df: pd.DataFrame):
    df["Entrance_timestamp"] = pd.to_datetime(df["Entrance_timestamp"])
    df = df.sort_values("Entrance_timestamp")

    # ------------------------------------------------------------
    # 1. Resample to hourly (fix missing hours)
    # ------------------------------------------------------------
    df_hourly = df.set_index("Entrance_timestamp").resample("20min").ffill()


    # ------------------------------------------------------------
    # 2. Compute daily smoothed weight (target)
    # ------------------------------------------------------------
    smooth_daily_weight = clean_daily_weights(
        df_hourly.reset_index(),
        weight_col="PREORE_VAKI-Weight [g]",
        date_col="Entrance_timestamp"
    )

    # Daily timestamps
    df_daily = df_hourly.resample("1D").first().iloc[:len(smooth_daily_weight)]
    df_daily["smooth_weight"] = smooth_daily_weight
    
    # Log-weight + log-growth
    df_daily["log_weight"] = np.log(df_daily["smooth_weight"].clip(1e-6))
    df_daily["log_growth"] = df_daily["log_weight"].diff().fillna(0.0)

    # ------------------------------------------------------------
    # 3. Prepare hourly features (X)
    # ------------------------------------------------------------
    feature_cols = [
        "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]",
        "PREORE_FEM_ENTRANCE-NH4+ [mg/L]",
        "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]",
        "PREORE_FEM_ENTRANCE-ODO [mg/L]",
        "PREORE_FEM_ENTRANCE-pH",
        "PREORE_FEM_ENTRANCE-Sal [psu]",
        "PREORE_FEM_ENTRANCE-Temp [Â°C]",
        # "I_Ration_Per_SamplingFrequency",
        "Energy_Acquisition(A)",
        "Catabolic_component(C)",
        "Somatic_tissue_energy_content(Epsilon)",
        "Feed_ration"
    ]

    df_hourly = df_hourly[feature_cols].copy()

    return df_hourly, df_daily

def create_windows_for_daily_targets(
        df_hourly,
        df_daily,
        feature_cols,
        history_hours=72,
        sampling_minutes=20):

    # How many samples per hour?
    samples_per_hour = int(60 / sampling_minutes)

    # Total number of rows in each window
    window_size = history_hours * samples_per_hour

    X, y = [], []
    daily_times = df_daily.index

    for i in range(1, len(daily_times)):
        day_t = daily_times[i]

        # Subtract EXACTLY 72 hours (biological window)
        start = day_t - pd.Timedelta(hours=history_hours)

        # Extract window of hourly data
        window = df_hourly.loc[start:day_t].iloc[:-1]

        # Skip incomplete windows
        if len(window) != window_size:
            continue

        X.append(window[feature_cols].values)
        y.append(df_daily["log_growth"].iloc[i])

    return np.array(X), np.array(y), window_size







# ============================================================
# TRAINING ON LOG-GROWTH
# ============================================================
def progressive_train(args, df_hourly, df_daily):
    
    fractions = [0.2, 0.5, 0.8]
    results = {}

    X_all, y_all, window_size = create_windows_for_daily_targets(
        df_hourly,
        df_daily,
        df_hourly.columns.tolist(),
        history_hours=args.history_hours,
        sampling_minutes=args.sampling_minutes
    )


    total = len(X_all)

    for frac in fractions:
        train_end = int(total * frac)

        print(f"\n=== Training with first {int(frac*100)}% ({train_end} samples) ===")

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]

        X_test = X_all[train_end:]
        y_test = y_all[train_end:]

        # Fit scaler only on training
        args.scaler_x.fit(X_train.reshape(-1, X_train.shape[-1]))

        X_train_scaled = np.array([args.scaler_x.transform(x) for x in X_train])
        X_test_scaled  = np.array([args.scaler_x.transform(x) for x in X_test])

        # Build model
        modelClass = ModelClass(window_size, X_train.shape[-1], args.dropout, args.learning_rate)
        model = modelClass.create_lstm_cnn_model()

        checkpointer = ModelCheckpoint(args.model_file, save_best_only=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

        model.fit(
            X_train_scaled, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2,
            callbacks=[checkpointer, early_stopping],
            verbose=args.verbose
        )

        # Predict log-growth
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()

        # Reconstruct weight
        start_idx = train_end + 1
        initial_log_weight = df_daily["log_weight"].iloc[start_idx]

        log_weight_pred = initial_log_weight + np.cumsum(y_pred)
        weight_pred = np.exp(log_weight_pred)

        weight_true = df_daily["smooth_weight"].iloc[start_idx:start_idx+len(weight_pred)].values

        mse, mae, mape = general.compute_metrics(weight_pred, weight_true, 0)

        results[f"{int(frac*100)}%"] = {
            "train_size": train_end,
            "test_size": len(X_test),
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "weight_pred": weight_pred,
            "weight_true": weight_true
        }
        
  



        

    return results







# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load dataset
    with open(args.root + "results/dynamic_individual_weight.pkl", "rb") as file:
        data = pickle.load(file)

    # Select campaign
    data_all = data[args.selected_campagin - 1]["df"]
    data_all = data_all.drop(
        ["index", "Unnamed: 0", "Exit_timestamp", "observed_timestamp"],
        axis=1,
        errors="ignore"
    )

    # Prepare daily data with log-growth
    df_hourly, df_daily = prepare_hourly_and_daily_data(data_all)
    


    # Build run name
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
    run_name = f"LogGrowth_{args.prediction_Method}_T{args.timesteps}_{formatted_datetime}"
    args.run_name = run_name
    args.path = f"data/Runs/{args.run_name}"
    args.model_file = args.path + "/fish_weight_loggrowth_model.keras"

    # Train model
    results = progressive_train(args, df_hourly, df_daily)
    print(results)
    
    
 

        
    
 




