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
import shap

# ============================================================
# ARGS
# ============================================================



@dataclass
class Args:
    # Model
    prediction_Method: str = "LSTM-CNN"

    # Training
    epochs: int = 80
    batch_size: int = 16
    patience: int = 8
    dropout: float = 0.2
    learning_rate: float = 0.001

    # Window length (hours)
    timesteps: int = 72     # 72-hour history

    # Cross-validation
    n_splits: int = 5       # IMPORTANT: only 3 folds for 30 days
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
    withTime:bool = True
    mathComputation= False
    
  

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
    
    if args.selected_campagin==3:
        df_daily_t= pd.read_csv("data/camp3_df_daily")
        df_daily["smooth_weight"] = df_daily_t["smooth_weight"]
    # df_daily.loc[df_daily.index[0], "smooth_weight"] = 25
    return df_daily["smooth_weight"].values


# ============================================================
# DAILY AGGREGATION + LOG-GROWTH TARGET
# ============================================================

def prepare_hourly_and_daily_data(df: pd.DataFrame):
    # ------------------------------------------------------------
    # 0. Timestamp + sorting
    # ------------------------------------------------------------
    df["Entrance_timestamp"] = pd.to_datetime(df["Entrance_timestamp"])
    df = df.sort_values("Entrance_timestamp")

    # ------------------------------------------------------------
    # 1. Resample to hourly (fix missing hours)
    # ------------------------------------------------------------
    df_hourly = df.set_index("Entrance_timestamp").resample("1h").ffill().bfill()

    # ------------------------------------------------------------
    # 2. Rolling feed-ration features (compute BEFORE df_daily)
    # ------------------------------------------------------------
    # 1h sampling → 24 samples/day
    df_hourly["Feed_ration_3d"] = (
        df_hourly["Feed_ration"].rolling(24 * 3).sum().bfill()
    )

    df_hourly["Feed_ration_7d"] = (
        df_hourly["Feed_ration"].rolling(24 * 7).sum().bfill()
    )
    
    df_hourly["temp_3d"] = (
        df_hourly["PREORE_FEM_ENTRANCE-Temp [Â°C]"].rolling(24 * 3).sum().bfill()
    )
    
    # In prepare_hourly_and_daily_data, add:
    for col in ["PREORE_FEM_ENTRANCE-Temp [Â°C]", "PREORE_FEM_ENTRANCE-ODO [mg/L]"]:
        df_hourly[f"delta_{col}"] = df_hourly[col].diff().fillna(0.0)
    

    # ------------------------------------------------------------
    # 3. Compute daily smoothed weight (target)
    # ------------------------------------------------------------
    smooth_daily_weight = clean_daily_weights(
        df_hourly.reset_index(),
        weight_col="PREORE_VAKI-Weight [g]",
        date_col="Entrance_timestamp"
    )
    
    if args.withTime:
        df_hourly["day"] = df_hourly.index.day
        df_hourly["month"] = df_hourly.index.month
        df_hourly["hour"] = df_hourly.index.hour

    df_daily = df_hourly.resample("1D").first().iloc[:len(smooth_daily_weight)]
    df_daily["smooth_weight"] = smooth_daily_weight
    

    # Log-weight
    # df_daily["log_weight"] = np.log(df_daily["smooth_weight"].clip(1e-6))
    df_daily["log_weight"] = np.log(df_daily["smooth_weight"].clip(1e-6))

    # Multiply log-growth by 10
    df_daily["log_growth"] = 1 * df_daily["log_weight"].diff().fillna(0.0)
    
    # ------------------------------------------------------------
    # 4. Select final feature set (remove raw Feed_ration)
    # ------------------------------------------------------------
    feature_cols = [
        "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]",
        "PREORE_FEM_ENTRANCE-NH4+ [mg/L]",
        "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]",
        "PREORE_FEM_ENTRANCE-ODO [mg/L]",
        "PREORE_FEM_ENTRANCE-pH",
        "PREORE_FEM_ENTRANCE-Sal [psu]",
        "PREORE_FEM_ENTRANCE-Temp [Â°C]",
        "Energy_Acquisition(A)",
        "Catabolic_component(C)",
        "Somatic_tissue_energy_content(Epsilon)",
        # 'I_Ration_Per_SamplingFrequency',
        "Feed_ration_3d",
        "Feed_ration",
        # "temp_3d"

    ]
    


    
    if args.reducedFeature:
        to_remove = ["Energy_Acquisition(A)",
        "Catabolic_component(C)",
        "Somatic_tissue_energy_content(Epsilon)"]
        feature_cols = [x for x in feature_cols if x not in to_remove]
    df_hourly = df_hourly[feature_cols].copy()

    return df_hourly, df_daily



def create_72h_windows_for_daily_targets(df_hourly, df_daily, feature_cols, window_hours=72):
    X, y = [], []

    daily_times = df_daily.index

    for i in range(1, len(daily_times)):
        day_t = daily_times[i]

        # Window start = 72 hours before day_t
        start = day_t - pd.Timedelta(hours=window_hours)
        if args.selected_campagin ==3:
            samples_per_hour = 60 // 20   # = 3
            window_size = 72 * samples_per_hour  # = 216
            
            window = df_hourly.loc[start:day_t].iloc[:-1]
            
            if len(window) != window_size:
                continue
        else:
            # Extract hourly window
            window = df_hourly.loc[start:day_t].iloc[:-1]  # exclude the exact day_t row
    
            if len(window) != window_hours:
                continue  # skip incomplete windows

        X.append(window[feature_cols].values)
        y.append(df_daily["log_growth"].iloc[i])

    return np.array(X), np.array(y)



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
def incremental_train(args, df_hourly, df_daily):
    writer = SummaryWriter(args.path)

    # ------------------------------------------------------------
    # 1. Build 72-hour windows for the entire dataset
    # ------------------------------------------------------------
    
    feature_cols = df_hourly.columns.tolist()
    
    if args.reducedFeature:
        remove_features = [ 'Feed_ration_7d',  'I_Ration_Per_SamplingFrequency', 'Energy_Acquisition(A)', 'Catabolic_component(C)', 'Somatic_tissue_energy_content(Epsilon)',    'delta_Energy_Acquisition(A)', 'delta_Catabolic_component(C)', 'delta_Somatic_tissue_energy_content(Epsilon)']
        feature_cols = [c for c in feature_cols if c not in remove_features]
    args.feature_names = feature_cols
    X_all, y_all = create_72h_windows_for_daily_targets(
        df_hourly, df_daily, feature_cols, window_hours=args.timesteps
    )

    total_days = len(X_all)
    chunk_size = total_days // args.num_splits

    fold_metrics = {}
    incremental_model = None

    # ------------------------------------------------------------
    # 2. Loop over incremental chunks
    # ------------------------------------------------------------
    for step in range(args.num_splits):
        print(f"\n=== Incremental Step {step+1}/{args.num_splits} ===")

        start = step * chunk_size
        end = (step+1) * chunk_size if step < args.num_splits-1 else total_days

        X_chunk = X_all[start:end]
        y_chunk = y_all[start:end]

        # ------------------------------------------------------------
        # Fit scaler ONLY on first chunk
        # ------------------------------------------------------------
        if step == 0:
            args.scaler_x.fit(X_chunk.reshape(-1, X_chunk.shape[-1]))

        # Scale chunk
        X_scaled = X_chunk.copy()
        for i in range(len(X_scaled)):
            X_scaled[i] = args.scaler_x.transform(X_scaled[i])

        # ------------------------------------------------------------
        # 3. TimeSeriesSplit inside each chunk
        # ------------------------------------------------------------
        tscv = TimeSeriesSplit(n_splits=args.n_splits)
        step_metrics = {"mse": [], "mae": [], "mape": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), start=1):
            print(f"  Fold {fold}/{args.n_splits}")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_chunk[train_idx], y_chunk[val_idx]

            # ------------------------------------------------------------
            # 4. Initialize or load model
            # ------------------------------------------------------------
            if incremental_model is None:
                modelClass = ModelClass(args.timesteps, X_train.shape[-1], args.dropout, args.learning_rate)
                
                if args.prediction_Method=="LSTM":
                    incremental_model = modelClass.create_LSTM()
                elif args.prediction_Method=="LSTM_CNN":
                    incremental_model = modelClass.create_lstm_cnn_model()
                elif args.prediction_Method=="CNN_LSTM":
                    incremental_model = modelClass.create_cnn_lstm_model()
                else:
                    incremental_model = modelClass.create_parallel_cnn_lstm_model()
                
            else:
                incremental_model.load_weights(args.model_file)

            checkpointer = ModelCheckpoint(args.model_file, save_best_only=True)
            early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

            incremental_model.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[checkpointer, early_stopping],
                verbose=args.verbose
            )

            # ------------------------------------------------------------
            # 5. Predict log-growth
            # ------------------------------------------------------------
            y_pred = incremental_model.predict(X_val, verbose=0).flatten()

            # ------------------------------------------------------------
            # 6. Reconstruct weight for evaluation
            # ------------------------------------------------------------
            global_val_start = start + val_idx[0]
            initial_log_weight = df_daily["log_weight"].iloc[global_val_start + 1]

            log_growth_true_scale = y_pred / 1.0

            log_weight_pred = initial_log_weight + np.cumsum(log_growth_true_scale)
            weight_pred = np.exp(log_weight_pred) # True weight for comparison weight_true = df_daily["smooth_weight"].iloc[ global_val_start + 1 : global_val_start + 1 + len(weight_pred) ].values

            weight_true = df_daily["smooth_weight"].iloc[global_val_start:global_val_start+len(weight_pred)].values

            mse, mae, mape = general.compute_metrics(weight_pred, weight_true, 0)
            step_metrics["mse"].append(mse)
            step_metrics["mae"].append(mae)
            step_metrics["mape"].append(mape)

        # Store metrics for this step
        fold_metrics[f"step_{step}"] = {
            "mse": np.mean(step_metrics["mse"]),
            "mae": np.mean(step_metrics["mae"]),
            "mape": np.mean(step_metrics["mape"]),
            "train_size": len(X_train),
            "val_size": len(X_val)
        }

    writer.close()
    return fold_metrics,step_metrics



# ============================================================
# PREDICT FULL WEIGHT TRAJECTORY
# ============================================================

def predict_full_weight(args: Args, df_hourly, df_daily):
    model = load_model(args.model_file)

    feature_cols = args.feature_names

    # ------------------------------------------------------------
    # 1. Build windows for full prediction
    # ------------------------------------------------------------
    X_raw, _ = create_72h_windows_for_daily_targets(
        df_hourly, df_daily, feature_cols, window_hours=args.timesteps
    )

    # ------------------------------------------------------------
    # 2. Scale windows
    # ------------------------------------------------------------
    X_scaled = X_raw.copy()
    for i in range(len(X_scaled)):
        X_scaled[i] = args.scaler_x.transform(X_scaled[i])

    # ------------------------------------------------------------
    # 3. Predict log-growth (scaled ×1)
    # ------------------------------------------------------------
    log_growth_pred_scaled = model.predict(X_scaled, verbose=0).flatten()

    # Undo scaling
    log_growth_pred = log_growth_pred_scaled / 1.0

    # ------------------------------------------------------------
    # 4. Reconstruct weight trajectory
    # ------------------------------------------------------------
    # X_raw[0] predicts df_daily[1]
    initial_log_weight = df_daily["log_weight"].iloc[1]

    log_weight_pred = initial_log_weight + np.cumsum(log_growth_pred)
    weight_pred = np.exp(log_weight_pred)

    # True weight for comparison
    weight_true = df_daily["smooth_weight"].iloc[1:1 + len(weight_pred)].values

    return weight_pred, weight_true


def perform_shap_analysis(args, df_hourly, df_daily):
    print("Initializing SHAP Analysis with KernelExplainer...")
    model = load_model(args.model_file)
    # 1. Prepare data
    X_raw, _ = create_72h_windows_for_daily_targets(
        df_hourly, df_daily, args.feature_names, window_hours=args.timesteps
    )
    
    X_scaled = X_raw.copy().astype('float32')
    for i in range(len(X_scaled)):
        X_scaled[i] = args.scaler_x.transform(X_scaled[i])
    
    # 2. Reshape for KernelExplainer (Samples, Timesteps * Features)
    # KernelExplainer handles 2D better; we will wrap the model to reshape it back
    n_samples, n_timesteps, n_features = X_scaled.shape
    X_flattened = X_scaled.reshape(n_samples, -1)
    
    # 3. Define a wrapper function to reshape data for the model
    def model_predict(data_2d):
        data_3d = data_2d.reshape(-1, n_timesteps, n_features)
        return model.predict(data_3d, verbose=0)

    # 4. Sampling
    # KernelExplainer is computationally heavy; use small samples
    background = shap.kmeans(X_flattened, 5) # Summarize background to 5 clusters
    test_size = min(10, n_samples)
    test_samples = X_flattened[np.random.choice(n_samples, test_size, replace=False)]

    # 5. Explain
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_samples)

    # shap_values is a list for multi-output; take index 0
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 6. Reshape SHAP values back to (samples, timesteps, features)
    shap_values_3d = shap_values.reshape(-1, n_timesteps, n_features)

    # 7. Global Importance (Bar Chart)
    global_importances = np.abs(shap_values_3d).mean(axis=(0, 1))
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(global_importances)
    plt.title("Global Feature Importance (KernelExplainer)")
    plt.barh(range(len(indices)), global_importances[indices], color='darkorange')
    plt.yticks(range(len(indices)), [args.feature_names[i] for i in indices])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(f"{args.path}/shap_kernel_bar.png")
    plt.show()

    importance_df = pd.DataFrame({
        'Feature': args.feature_names,
        'Mean_Abs_SHAP': global_importances
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"global_feature_importance_modelfree_{timestamp}.csv"
    importance_df.to_csv(f"data/{filename}", index=False)
    
    print("SHAP data successfully saved ")




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.selected_campagin==3:
        args.timesteps = 72*3
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

    fold_metrics,step_metrics = incremental_train(args, df_hourly, df_daily)
    metrics = ["mse", "mae", "mape"]
    print (f"================================{args.prediction_Method}===================================")
    if args.num_splits==1:
        mean_metrics = {}
        std_metrics = {}
        
        for metric, values in step_metrics.items():
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values, ddof=1)  # sample std
        
        print("Mean per metric:", mean_metrics)
        print("Std per metric:", std_metrics)
    else:
        print(fold_metrics)



    # Predict full trajectory
    weight_pred, weight_true = predict_full_weight(args, df_hourly, df_daily)



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
    
    if args.mathComputation:
    
        
        
        math_weight = df_daily["mathematical_computed_weight"].iloc[1:1+len(weight_pred)].values
       
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
    perform_shap_analysis(args, df_hourly, df_daily)
   
    
    
    # # # Select campaign
    # data_all = data[0]["df"].copy()
    # df = data_all.drop(
    #     ["index", "Unnamed: 0", "Exit_timestamp", "observed_timestamp"],
    #     axis=1,
    #     errors="ignore"
    # )

    # # Prepare daily data with log-growth
    # # df_hourly, df_daily = prepare_hourly_and_daily_data(data_all)
    


    # # df_daily_t= pd.read_csv("data/camp3_df_daily")
    # # df_daily["smooth_weight"] = df_daily_t["smooth_weight"].values

    # df["Entrance_timestamp"] = pd.to_datetime(df["Entrance_timestamp"])
    # df = df.sort_values("Entrance_timestamp")

    # # ------------------------------------------------------------
    # # 1. Resample to hourly (fix missing hours)
    # # ------------------------------------------------------------
    # df_hourly = df.set_index("Entrance_timestamp").resample("1h").ffill().bfill()

    # # ------------------------------------------------------------
    # # 2. Rolling feed-ration features (compute BEFORE df_daily)
    # # ------------------------------------------------------------
    # # 1h sampling → 24 samples/day
    # df_hourly["Feed_ration_3d"] = (
    #     df_hourly["Feed_ration"].rolling(24 * 3).sum().bfill()
    # )

    # df_hourly["Feed_ration_7d"] = (
    #     df_hourly["Feed_ration"].rolling(24 * 7).sum().bfill()
    # )
    
    # df_hourly["temp_3d"] = (
    #     df_hourly["PREORE_FEM_ENTRANCE-Temp [Â°C]"].rolling(24 * 3).sum().bfill()
    # )
    
    # # In prepare_hourly_and_daily_data, add:
    # for col in ["PREORE_FEM_ENTRANCE-Temp [Â°C]", "PREORE_FEM_ENTRANCE-ODO [mg/L]"]:
    #     df_hourly[f"delta_{col}"] = df_hourly[col].diff().fillna(0.0)
    

    # # ------------------------------------------------------------
    # # 3. Compute daily smoothed weight (target)
    # # ------------------------------------------------------------
    # smooth_daily_weight = clean_daily_weights(
    #     df_hourly.reset_index(),
    #     weight_col="PREORE_VAKI-Weight [g]",
    #     date_col="Entrance_timestamp"
    # )
    # # df_daily_t= pd.read_csv("data/camp3_df_daily")
    # # smooth_daily_weight = df_daily_t["smooth_weight"].values
    # df_daily = df_hourly.resample("1D").first().iloc[:len(smooth_daily_weight)]
    # df_daily["smooth_weight"] = smooth_daily_weight
    

    # # Log-weight
    # # df_daily["log_weight"] = np.log(df_daily["smooth_weight"].clip(1e-6))
    # df_daily["log_weight"] = np.log(df_daily["smooth_weight"].clip(1e-6))

    # # Multiply log-growth by 10
    # df_daily["log_growth"] = 1 * df_daily["log_weight"].diff().fillna(0.0)

    # # ------------------------------------------------------------
    # # 4. Select final feature set (remove raw Feed_ration)
    # # ------------------------------------------------------------
    # feature_cols = [
    #     "PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]",
    #     "PREORE_FEM_ENTRANCE-NH4+ [mg/L]",
    #     "PREORE_FEM_ENTRANCE-NO3 -N [mg/L]",
    #     "PREORE_FEM_ENTRANCE-ODO [mg/L]",
    #     "PREORE_FEM_ENTRANCE-pH",
    #     "PREORE_FEM_ENTRANCE-Sal [psu]",
    #     "PREORE_FEM_ENTRANCE-Temp [Â°C]",
    #     "Energy_Acquisition(A)",
    #     "Catabolic_component(C)",
    #     "Somatic_tissue_energy_content(Epsilon)",
    #     'I_Ration_Per_SamplingFrequency',
    #     "Feed_ration_3d",
    #     "Feed_ration",
    #     # "temp_3d"

    # ]

    # df_hourly = df_hourly[feature_cols].copy()


    # # # Predict full trajectory
    # weight_pred, weight_true = predict_full_weight(args, df_hourly, df_daily)



    # # # Log plots
    # writer = SummaryWriter(args.path)
    # plots = Custom_plots(
    #     weight_pred.reshape(-1, 1),
    #     weight_true.reshape(-1, 1),
    #     writer,
    #     title="Predicted vs True Weight (Daily, Log-Growth Reconstruction)",
    #     summarytitle=args.run_name
    # )
    # plots.plot_all()
    # writer.close()
    
    # mse, mae, mape = general.compute_metrics(weight_pred, weight_true)
    
    # # math_weight = df_daily["mathematical_computed_weight"].iloc[1:1+len(weight_pred)].values
   
    # # mse, mae, mape = general.compute_metrics(math_weight, weight_true)
    
    # # plots = Custom_plots(
    # #     math_weight.reshape(-1, 1),
    # #     weight_true.reshape(-1, 1),
    # #     writer,
    # #     title="Predicted vs True Weight (Daily, Log-Growth Reconstruction)",
    # #     summarytitle=args.run_name
    # # )
    # # plots.plot_all()
    # # writer.close()

