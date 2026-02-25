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
class Args:
    # "LSTM", "LSTM_CNN", "CNN_LSTM", "Parrarel_CNN_LSTM"
    prediction_Method: str = "Parrarel_CNN_LSTM"

    # RNN hyperparams
    epochs: int = 80
    batch_size: int = 32
    timesteps: int = 7          # 7 days of history
    patience: int = 10
    dropout: float = 0.2
    learning_rate: float = 0.001
    n_splits: int = 5
    verbose: int = 0

    # Preprocessing
    scale_flag: bool = True

    # Paths and run info
    root: str = "data/Preore_Dataset/"
    path: str = ""
    model_file: str = ""
    run_name: str = ""
    selected_campagin: int = 2

    # Internals
    feature_names: list[str] | None = None
    scaler_x: MinMaxScaler = MinMaxScaler()
    reducedFeature: bool = False  
    
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


def _clean_daily_weights(df, weight_col='PREORE_VAKI-Weight [g]', date_col='Entrance_timestamp'):
    # Ensure date is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    df['date_only'] = df[date_col].dt.date
    
    cleaned_data = []

    # Process each day individually
    for date, group in df.groupby('date_only'):
        group = group.sort_values(by=date_col)
        
        # 1. Identify the thresholds for the day
        low_threshold = group[weight_col].quantile(0.25)
        high_threshold = group[weight_col].quantile(0.75)
        
        # 2. Mark outliers as NaN
        # Values outside the 10th-90th percentile are cleared
        group.loc[(group[weight_col] < low_threshold) | 
                  (group[weight_col] > high_threshold), weight_col] = np.nan
        
        # 3. Interpolate the NaNs using valid readings from the same day
        # 'limit_direction' ensures we fill both start and end of the day
        group[weight_col] = group[weight_col].interpolate(method='linear', limit_direction='both')
        
        cleaned_data.append(group)
    
    # Reconstruct the dataframe
    df_cleaned = pd.concat(cleaned_data).drop(columns=['date_only'])
    
    # 4. Final safety: If a whole day was noisy and resulted in NaNs, 
    # fill them using the global trend
    df_cleaned[weight_col] = df_cleaned[weight_col].ffill().bfill()
    
    series = df_cleaned[weight_col]
    df_cleaned[weight_col] = savgol_filter(series, window_length=5, polyorder=2)
    
    plt.plot(df["PREORE_VAKI-Weight [g]"], marker='o', linestyle='-', color='blue')  # line plot with marker

    # Save initial weight (first day mean)
    # initial_weight = df_cleaned[weight_col].iloc[0]

    

    # Add day counter (0,1,2,...)
    # df_cleaned['day'] = np.arange(len(df_cleaned))

    # Add initial weight column
    # df_cleaned['initial_weight'] = initial_weight

    return df_cleaned
# ============================================================
# DAILY AGGREGATION + LOG-GROWTH TARGET
# ============================================================

def prepare_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Entrance_timestamp"] = pd.to_datetime(df["Entrance_timestamp"])
    df = df.drop(['PREORE_FEM_ENTRANCE-Depth [m]','PREORE_FEM_ENTRANCE-NH3 [mg/L]'],axis = 1)
    
    df = df.sort_values("Entrance_timestamp")
    smooth_weight = clean_daily_weights(df, weight_col='PREORE_VAKI-Weight [g]', date_col='Entrance_timestamp')
    # Daily aggregation
    df_daily = df.resample("D", on="Entrance_timestamp").agg({
        
        "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]": "mean",
        
        "PREORE_FEM_ENTRANCE-NH4+ [mg/L]": "mean",
        "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]": "mean",
        "PREORE_FEM_ENTRANCE-ODO [mg/L]": "mean",
        "PREORE_FEM_ENTRANCE-pH": "mean",
        "PREORE_FEM_ENTRANCE-Sal [psu]": "mean",
        "PREORE_FEM_ENTRANCE-Temp [Â°C]": "mean",
        "I_Ration_Per_SamplingFrequency": "sum",
        "Energy_Acquisition(A)": "mean",
        "Catabolic_component(C)": "mean",
        "Somatic_tissue_energy_content(Epsilon)": "mean",
        "Feed_ration": "sum",
        "mathematical_computed_weight": "mean"
    }).dropna()
    df_daily["PREORE_VAKI-Weight [g]"] = smooth_weight
    # Log-weight
    df_daily["log_weight"] = np.log(df_daily["PREORE_VAKI-Weight [g]"].clip(lower=1e-6))

    # Log-growth target: Δ log(weight)
    df_daily["log_growth"] = df_daily["log_weight"].diff().fillna(0.0)

    # Environmental deltas (day-to-day changes)
    env_cols = [
        "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]",

        "PREORE_FEM_ENTRANCE-NH4+ [mg/L]",
        "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]",
        "PREORE_FEM_ENTRANCE-ODO [mg/L]",
        "PREORE_FEM_ENTRANCE-pH",
        "PREORE_FEM_ENTRANCE-Sal [psu]",
        "PREORE_FEM_ENTRANCE-Temp [Â°C]",
        "Energy_Acquisition(A)",
        "Catabolic_component(C)",
        "Somatic_tissue_energy_content(Epsilon)"
    ]
    for col in env_cols:
        if col in df_daily.columns:
            df_daily[f"delta_{col}"] = df_daily[col].diff().fillna(0.0)

    # Rolling ration features
    if "Feed_ration" in df_daily.columns:
        df_daily["Feed_ration_3d"] = df_daily["Feed_ration"].rolling(3, min_periods=1).sum()
        df_daily["Feed_ration_7d"] = df_daily["Feed_ration"].rolling(7, min_periods=1).sum()

    return df_daily


# ============================================================
# SLIDING WINDOWS ON DAILY DATA
# ============================================================

def create_daily_windows(df_daily: pd.DataFrame, timesteps: int, feature_cols: list):
    features = df_daily[feature_cols].values
    target = df_daily["log_growth"].values

    X, y = [], []
    for i in range(len(df_daily) - timesteps):
        X.append(features[i:i+timesteps])
        y.append(target[i+timesteps])
    return np.array(X), np.array(y)


# ============================================================
# TRAINING ON LOG-GROWTH
# ============================================================

def train_growth_model(args: Args, df_daily: pd.DataFrame):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "Hyperparameters",
        "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    )

    # Define feature columns (exclude targets)
    base_cols = [
        "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]",

        "PREORE_FEM_ENTRANCE-NH4+ [mg/L]",
        "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]",
        "PREORE_FEM_ENTRANCE-ODO [mg/L]",
        "PREORE_FEM_ENTRANCE-pH",
        "PREORE_FEM_ENTRANCE-Sal [psu]",
        "PREORE_FEM_ENTRANCE-Temp [Â°C]",
        "I_Ration_Per_SamplingFrequency",
        "Energy_Acquisition(A)",
        "Catabolic_component(C)",
        "Somatic_tissue_energy_content(Epsilon)",
        "Feed_ration"
    ]
    delta_cols = [c for c in df_daily.columns if c.startswith("delta_")]
    ration_cols = [c for c in df_daily.columns if c.startswith("Feed_ration_")]

    feature_cols = [c for c in base_cols if c in df_daily.columns] + delta_cols + ration_cols
    args.feature_names = feature_cols

    X_raw = df_daily[feature_cols].values
    y_raw = df_daily["log_growth"].values

    # Scale features
    if args.scale_flag:
        args.scaler_x.fit(X_raw)
        X_scaled = args.scaler_x.transform(X_raw)
    else:
        X_scaled = X_raw

    df_scaled = df_daily.copy()
    df_scaled[feature_cols] = X_scaled

    # Sliding windows
    X_seq, y_seq = create_daily_windows(df_scaled, args.timesteps, feature_cols)

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    fold_metrics = {"mse": [], "mae": [], "mape": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq), start=1):
        print(f"Training fold {fold}/{args.n_splits}")

        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_seq[train_idx], y_seq[val_idx]

        modelClass = ModelClass(args.timesteps, X_train.shape[-1], args.dropout, args.learning_rate)
        if args.prediction_Method == "LSTM":
            model = modelClass.create_LSTM()
        elif args.prediction_Method == "LSTM_CNN":
            model = modelClass.create_lstm_cnn_model()
        elif args.prediction_Method == "CNN_LSTM":
            model = modelClass.create_cnn_lstm_model()
        else:
            model = modelClass.create_parallel_cnn_lstm_model()

        checkpointer = ModelCheckpoint(args.model_file, save_best_only=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpointer, early_stopping],
            verbose=args.verbose
        )

        y_pred = model.predict(X_val, verbose=0).flatten()
        mse, mae, mape = general.compute_metrics(y_pred, y_val,0)
        fold_metrics["mse"].append(mse)
        fold_metrics["mae"].append(mae)
        fold_metrics["mape"].append(mape)

        table = (
            f"| Fold | MSE | MAE | MAPE |\n|-|-|-|-|\n"
            f"| {fold} | {mse:.6f} | {mae:.6f} | {mape:.6f} |"
        )
        writer.add_text(f"Metrics_Fold_{fold}", table)

    avg_mse = np.mean(fold_metrics["mse"])
    avg_mae = np.mean(fold_metrics["mae"])
    avg_mape = np.mean(fold_metrics["mape"])
    writer.add_text(
        "Final_CV_Metrics",
        f"MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, MAPE: {avg_mape:.6f}"
    )
    print(f"Final CV - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, MAPE: {avg_mape:.6f}")

    writer.close()
    return


# ============================================================
# PREDICT FULL WEIGHT TRAJECTORY
# ============================================================

def predict_full_weight(args: Args, df_daily: pd.DataFrame):
    model = load_model(args.model_file)

    feature_cols = args.feature_names
    X_raw = df_daily[feature_cols].values

    if args.scale_flag:
        X_scaled = args.scaler_x.transform(X_raw)
    else:
        X_scaled = X_raw

    df_scaled = df_daily.copy()
    df_scaled[feature_cols] = X_scaled

    X_seq, y_seq = create_daily_windows(df_scaled, args.timesteps, feature_cols)

    # Predict log-growth
    log_growth_pred = model.predict(X_seq, verbose=0).flatten()

    # Reconstruct log-weight and weight
    # First prediction corresponds to index = timesteps
    initial_log_weight = df_daily["log_weight"].iloc[args.timesteps]
    log_weight_pred = initial_log_weight + np.cumsum(log_growth_pred)
    weight_pred = np.exp(log_weight_pred)

    # True weight aligned
    true_weight = df_daily["PREORE_VAKI-Weight [g]"].iloc[args.timesteps:args.timesteps+len(weight_pred)].values
    
    
    return weight_pred, true_weight


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
    df_daily = prepare_daily_data(data_all)

    # Build run name
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
    run_name = f"LogGrowth_{args.prediction_Method}_T{args.timesteps}_{formatted_datetime}"
    args.run_name = run_name
    args.path = f"data/Runs/{args.run_name}"
    args.model_file = args.path + "/fish_weight_loggrowth_model.keras"

    # Train model
    train_growth_model(args, df_daily)

    # Predict full trajectory
    weight_pred, weight_true = predict_full_weight(args, df_daily)

    # Log plots
    writer = SummaryWriter(args.path)
    plots = Custom_plots(
        weight_pred.reshape(-1, 1),
        weight_true.reshape(-1, 1),
        writer,
        title="Predicted vs True Weight (Daily, Log-Growth Reconstruction)",
        summarytitle=args.run_name
    )
    plots.plot_all()
    writer.close()
    
    mse, mae, mape = general.compute_metrics(weight_pred, weight_true)
    
    math_weight = df_daily["mathematical_computed_weight"].iloc[args.timesteps:args.timesteps+len(weight_pred)].values
    mse, mae, mape = general.compute_metrics(math_weight, weight_true)
    
    plots = Custom_plots(
        math_weight.reshape(-1, 1),
        weight_true.reshape(-1, 1),
        writer,
        title="Predicted vs True Weight (Daily, Log-Growth Reconstruction)",
        summarytitle=args.run_name
    )
    plots.plot_all()
    writer.close()
