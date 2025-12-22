from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # high-level visualization based on matplotlib
import pickle
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Custom_plots import Custom_plots
from ModelClass import ModelClass
from sklearn.model_selection import TimeSeriesSplit
from general import general
from rainbow_trout_model import rainbow_trout_model
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="LSTM_CNN" 
    
    if prediction_Method!="Random_Forest":
        verbos= 0
        epochs: int= 50
        batch_size: int= 32
        validation_split: float= 0.2
        timesteps: int= 1
        patience: int= 10
        dropout: float= 0.2 
        learning_rate: float= 0.001 
        n_splits= 5
        verbose= 0
        scale_flag: bool() = True
        transformFlag: bool() = True

    
    

    withTime: bool() = True
    reducedFeature: bool() = True    
    root = 'data/Preore_Dataset/'
    path=""
    model_file = ""
    displayCorrMatrix = False
    feature_names = []
    # Scalers
    scaler_x = MinMaxScaler()
    time_windows_size = []
    time_windows = []
    run_name=""
    selected_time_windows = 2
    data_augmentation = True




# Log the training and validation metrics to TensorBoard
def log_metrics(writer, history):
    for epoch in range(len(history.history['loss'])):
        writer.add_scalar("Loss/Train", history.history['loss'][epoch], epoch)
        writer.add_scalar("Loss/Val", history.history['val_loss'][epoch], epoch)
        writer.add_scalar("MSE/Train", history.history['mse'][epoch], epoch)
        writer.add_scalar("MSE/Val", history.history['val_mse'][epoch], epoch)
        writer.add_scalar("MAE/Train", history.history['mae'][epoch], epoch)
        writer.add_scalar("MAE/Val", history.history['val_mae'][epoch], epoch)
        writer.add_scalar("MAPE/Train", history.history['mape'][epoch], epoch)
        writer.add_scalar("MAPE/Val", history.history['val_mape'][epoch], epoch)

def compute_mathematical_weights(args,future_df): 
    rainbow_trout = rainbow_trout_model(data[args.selected_time_windows]['sampling_rate_per_day'])
    W0 = future_df['PREORE_VAKI-Weight [g]'][0]
    time_steps = np.arange(len(future_df))
    t_span = (0, len(time_steps))
    t_eval = np.linspace(0, len(time_steps), len(time_steps))
    temperature_data = future_df['PREORE_FEM_ENTRANCE-Temp [Â°C]'].to_numpy()
    temperature_app = interp1d(time_steps, temperature_data, kind='linear', fill_value="extrapolate")
    future_df['I_Ration_Per_SamplingFrequency'] = future_df.apply(lambda row: rainbow_trout.Input_ration(row['PREORE_VAKI-Weight [g]'], row['PREORE_FEM_ENTRANCE-Temp [Â°C]']),axis=1)
    g_app = interp1d(time_steps, future_df['I_Ration_Per_SamplingFrequency'], kind='linear', fill_value="extrapolate")
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
        
    future_df['Energy_Acquisition(A)']=Energy_Acquisition
    future_df['Catabolic_component(C)']=Catabolic_component
    future_df['Somatic_tissue_energy_content(Epsilon)']=Somatic_tissue_energy_content
    future_df['mathematical_computed_weight']=solution.y[0]
    future_df['Feed_ration']=Feed_ration
    
    return future_df
      
def extend_data(args,df):
    df = df.rename(columns={'day_of_month': 'day'})

    
    # Create new timestamps (e.g., for the next 30 days hourly)
    if args.selected_time_windows==2:
        # Assume your existing DataFrame is named `df`
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour','minute']])
        
        # Find the latest timestamp
        last_date = df['datetime'].max()
        start_time = last_date + pd.Timedelta(minutes=20)

        # Generate future dates every 20 minutes for 30 days (30 * 24 * 3 = 2160 timestamps)
        future_dates = pd.date_range(start=start_time, periods=30*24*3, freq='20min')
        # Extract date parts
        future_df = pd.DataFrame({
            'datetime': future_dates,
            'year': future_dates.year,
            'month': future_dates.month,
            'day_of_month': future_dates.day,
            'hour': future_dates.hour,
            'minute': future_dates.minute
        })
    else:
        # Assume your existing DataFrame is named `df`
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # Find the latest timestamp
        last_date = df['datetime'].max()
        

        # Generate future dates every 20 minutes for 30 days (30 * 24 * 3 = 2160 timestamps)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=30*24, freq='h')
    
    
        # Extract date parts
        future_df = pd.DataFrame({
            'datetime': future_dates,
            'year': future_dates.year,
            'month': future_dates.month,
            'day_of_month': future_dates.day,
            'hour': future_dates.hour
        })
    # Choose the features to generate
    features_to_generate = [
        'PREORE_FEM_ENTRANCE-Cond [ÂµS/cm]',
        'PREORE_FEM_ENTRANCE-Depth [m]',
        'PREORE_FEM_ENTRANCE-NH3 [mg/L]',
        'PREORE_FEM_ENTRANCE-NH4+ [mg/L]',
        'PREORE_FEM_ENTRANCE-NO3 -N [mg/L]',
        'PREORE_FEM_ENTRANCE-ODO [mg/L]',
        'PREORE_FEM_ENTRANCE-pH',
        'PREORE_FEM_ENTRANCE-Sal [psu]',
        'PREORE_FEM_ENTRANCE-Temp [Â°C]',
        'PREORE_VAKI-Weight [g]'

    ]
    
    # Fit simple models or compute mean and std by hour
    hourly_stats = df.groupby('hour')[features_to_generate].agg(['mean', 'std'])
    
    # Fill synthetic values in the new DataFrame
    for col in features_to_generate:
        means = future_df['hour'].map(hourly_stats[col]['mean'])
        stds = future_df['hour'].map(hourly_stats[col]['std'])
        noise = np.random.normal(loc=0, scale=1, size=len(future_df))
        future_df[col] = means + stds * noise
    
    future_df = future_df.drop(columns='datetime')
    
    future_df = compute_mathematical_weights(args,future_df)
    return future_df
    
def prepare_data(args, data):
    # Drop unwanted columns
    data = data.drop(data.columns[data.columns.str.contains('EXIT')], axis=1)
    data['Entrance_timestamp'] = pd.to_datetime(data['Entrance_timestamp'])
    data = data.drop(["PREORE_VAKI-Length [mm]", "PREORE_VAKI-CF"], axis=1)
    
    # Extract date parts if required
    if args.withTime:
        data['year'] = data['Entrance_timestamp'].dt.year
        data['month'] = data['Entrance_timestamp'].dt.month
        data['day_of_month'] = data['Entrance_timestamp'].dt.day
        data['hour'] = data['Entrance_timestamp'].dt.hour
        if args.selected_time_windows==2:
            data['minute'] = data['Entrance_timestamp'].dt.minute
    
    # Drop the original timestamp column
    data = data.drop(["Entrance_timestamp"], axis=1)

    # Drop additional columns based on reduced feature flag
    
    if args.data_augmentation:
        future_data = extend_data(args,data)
        data = pd.concat([data, future_data], ignore_index=True)
        if args.selected_time_windows==2:
            data = data.sort_values(by=['year', 'month', 'day_of_month','hour','minute']).reset_index(drop=True)
        else:
            data = data.sort_values(by=['year', 'month', 'day_of_month','hour']).reset_index(drop=True)
            
    if args.reducedFeature:
        data = data.drop(["Energy_Acquisition(A)", "Catabolic_component(C)", "Somatic_tissue_energy_content(Epsilon)","I_Ration_Per_SamplingFrequency"], axis=1)
    # Separate target and feature columns
    y = data["PREORE_VAKI-Weight [g]"].values
    x = data.drop(["PREORE_VAKI-Weight [g]", "mathematical_computed_weight"], axis=1)
    
    
    args.feature_names = x.columns

    # Apply transformation to target if flag is set
    if args.transformFlag:
        y = np.log1p(y)  # Use log1p for stability
        
    
    # Separate date-related features if scaling is applied
    if args.scale_flag:
        x = args.scaler_x.fit_transform(x)
    else:
        x = x.values  # Keep the original values if scaling isn't applied
    
    def fill_nan_with_previous(arr):
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                arr[i] = arr[i-1]
        return arr

    # Example:
    
    y = fill_nan_with_previous(y.copy())
    x = np.array(x)
    return x, y, data


def split_data(x,y,data_contained_fishWeight):
    # Step 1: Train-Test Split
    indices = np.arange(len(x))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=0.2, random_state=23)
    
    original_x_test = x_test.copy()
    original_y_test = y_test.copy()
    
    if args.prediction_Method!="Random_Forest": 
        # Step 2: Reshape to sequences after train-test split
        samples_train = int(x_train.shape[0] / args.timesteps)
        x_train = x_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps, x_train.shape[1])
        y_train = y_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps)
        
        samples_test = int(x_test.shape[0] / args.timesteps)
        x_test = x_test[:samples_test * args.timesteps].reshape(samples_test, args.timesteps, x_test.shape[1])
        y_test = y_test[:samples_test * args.timesteps].reshape(samples_test, args.timesteps)
    
    # Step 3: If you need to get back to the original 'Fish_Weight' values
    Fish_Weight_Predictedby_Math_model = np.array(data_contained_fishWeight.iloc[idx_test]["Fish_Weight"]).reshape(-1,1)
    data = data_contained_fishWeight.drop(columns=["Fish_Weight"])
    return x_train, x_test, y_train, y_test,data, Fish_Weight_Predictedby_Math_model, original_x_test, original_y_test



    
  
def corr_matrix(data, writer):
    # Compute the correlation matrix
    correMtr = data.corr()
    mask = np.array(correMtr)
    mask[np.tril_indices_from(mask)] = False  # the correlation matrix is symmetric

    # Prepare the weight correlations for the bar chart
    weight_corr = correMtr['PREORE_VAKI-Weight [g]'].drop('PREORE_VAKI-Weight [g]').sort_values()

    # Create a figure with 2 subplots (1x2 grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot the correlation heatmap on the first axis
    sns.set_style("white")
    sns.heatmap(correMtr, mask=mask, vmin=-1.0, vmax=1.0, square=True, annot=True, fmt=".2f", ax=ax1)
    ax1.set_title('Correlation matrix of attributes')
    
    # Plot the vertical bar chart on the second axis
    sns.barplot(y=weight_corr.index, x=weight_corr.values,hue=weight_corr.index, legend=False, ax=ax2)
    ax2.set_xlabel("Correlation with PREORE_VAKI-Weight [g]")
    ax2.set_ylabel("Features")
    ax2.set_title("Correlation of Features with PREORE_VAKI-Weight [g]")
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()
    # Log the combined figure to TensorBoard
    writer.add_figure("Combined Correlation Plots", fig)
    





def assign_labels(val_index, boundaries):
    labels = np.zeros_like(val_index, dtype=int)
    for i, (start, end) in enumerate(boundaries):
        labels[(val_index >= start) & (val_index <= end)] = i + 1
    return labels

def train(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Convert time window sizes into ranges
    ranges = [0] + args.time_windows_size  # Add a starting point at 0
    ranges = np.cumsum(ranges)
    boundaries = [(ranges[i], ranges[i+1]-1) for i in range(len(args.time_windows_size))]

    x, y, data_contained_fishWeight = prepare_data(args, data)
    # x_train, x_test, y_train, y_test,data,Fish_Weight_Predictedby_Math_model, original_x_test, original_y_test= split_data( x, y, data_contained_fishWeight)
    data = data_contained_fishWeight.drop(columns=["mathematical_computed_weight"])
    
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in args.feature_names])
    writer.add_text("3: Feature Names", feature_table)

    
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
    
    # kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=23)
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = {"mse": [],"mae": [],"mape": []}
   
    for fold, (train_index, val_index) in enumerate(tscv.split(x)):
        print(f"Training fold {fold + 1}/{args.n_splits}")                                                                
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        labels = assign_labels(val_index, boundaries)
        train_labels = assign_labels(train_index, boundaries)
        
        Fish_Weight_Predictedby_Math_model = np.array(data_contained_fishWeight.iloc[val_index]["mathematical_computed_weight"]).reshape(-1,1)
        
        original_x_test = x_val.copy()
        original_y_test = y_val.copy()
        
        # Step 2: Reshape to sequences after train-test split
        samples_train = int(x_train.shape[0] / args.timesteps)
        x_train = x_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps, x_train.shape[1])
        y_train = y_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps)
        
        samples_test = int(x_val.shape[0] / args.timesteps)
        x_val = x_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps, x_val.shape[1])
        y_val = y_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps)
        
        
        modelClass = ModelClass(args.timesteps, x_train.shape[2], args.dropout, args.learning_rate)
        
        if args.prediction_Method=="LSTM":
            model = modelClass.create_LSTM()
        elif args.prediction_Method=="LSTM_CNN":
            model = modelClass.create_lstm_cnn_model()
        elif args.prediction_Method=="CNN_LSTM":
            model = modelClass.create_cnn_lstm_model()
        else:
            model = modelClass.create_parallel_cnn_lstm_model()
        
        # model = modelClass.create_LSTM()
         
        checkpointer = ModelCheckpoint(filepath=args.model_file, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
        
        if fold + 1!=args.n_splits:
            history = model.fit(
                x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                validation_split=args.validation_split, callbacks=[early_stopping], verbose=args.verbos)
        else:
            history = model.fit(
                x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                validation_split=args.validation_split, callbacks=[checkpointer, early_stopping], verbose=args.verbos)
        
            log_metrics(writer, history)
            for metric in ['mse', 'mae', 'mape']:
                fig, ax = plt.subplots()
                ax.plot(history.history[metric], label=f'Training {metric.upper()}')
                ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
                ax.set_xlabel('Epochs')
                ax.set_ylabel(metric.upper())
                ax.legend()
                writer.add_figure(f"Training/{metric.upper()}", fig)
                plt.close(fig)  
    

        # test_results = model.evaluate(x_val, y_val, verbose=args.verbos)
        
        # Ensure inputs are 3D by adding a dimension if necessary

        if original_x_test.ndim == 2:
            reshaped_inputs = np.tile(original_x_test[:, np.newaxis, :], (1, args.timesteps, 1))
        else:  # Already 3D
            reshaped_inputs = np.tile(original_x_test, (1, args.timesteps, 1))
        
        # Batch prediction
        y_preds = model.predict(reshaped_inputs, batch_size=32)  # Adjust batch_size for your hardware
        
        # Post-process predictions and actual values
        if args.transformFlag:
            predicted_values = np.expm1(y_preds[:, 0])  # Transform predicted values
            actual_values = np.expm1(original_y_test)  # Transform ground truth
        else:
            predicted_values = y_preds[:, 0]  # Directly take predictions
            actual_values = original_y_test  # Directly use ground truth
        
        # Reshape to required dimensions
        predicted_values = predicted_values.reshape(-1, 1)
        actual_values = actual_values.reshape(-1, 1)
        

        mse1,mae1,mape1= general.compute_metrics(predicted_values, actual_values)
        fold_metrics["mse"].append(mse1)
        fold_metrics["mae"].append(mae1)
        fold_metrics["mape"].append(mape1)
        
        if fold + 1==args.n_splits:
            mse1= np.mean(fold_metrics["mse"])
            mae1= np.mean(fold_metrics["mae"])
            mape1= np.mean(fold_metrics["mape"])
            for p in np.unique(labels):
                title = "Time window: " +str(p)+ args.time_windows[p-1] + "  (Train size: "+str(len(train_labels[train_labels==p]))+")"
                plots = Custom_plots(predicted_values[labels==p],actual_values[labels==p],writer=writer,title=title,summarytitle=args.run_name+"(TimeWindow: "+str(p)+")")
                plots.plot_all()
                plt.close(fig)
            print("*****************************************************************************************")  

            mse2,mae2,mape2= general.compute_metrics(Fish_Weight_Predictedby_Math_model, actual_values)
            table_header = f"| Metric | {args.prediction_Method} | Method based on mathematical model |\n|-|-|-|"
            table_rows = f"| MSE   | {mse1:.4f} | {mse2:.4f} |\n"
            table_rows += f"| MAE   | {mae1:.4f} | {mae2:.4f} |\n"
            table_rows += f"| MAPE  | {mape1:.4f} | {mape2:.4f} |"
            table = f"{table_header}\n{table_rows}"
            writer.add_text("1: Metrics Comparison", table)
            print(table)
            for p in np.unique(labels):
                title = "Method based on mathematical model\nTime window: " +str(p)+ args.time_windows[p-1]
                plots = Custom_plots(Fish_Weight_Predictedby_Math_model[labels==p],actual_values[labels==p],writer=writer, title=title, summarytitle="Method based on mathematical model(TimeWindow: "+str(p)+")")
                plots.plot_all()
                plt.close(fig)
            # plots = Custom_plots(Fish_Weight_Predictedby_Math_model, actual_values,writer,"Method based on mathematical model_")
            # plots.plot_all()
            # plt.close(fig)
            writer.close()
            
            
            
            
            #=======================Save predicted results for all data
            if original_x_test.ndim == 2:
                reshaped_inputs = np.tile(x[:, np.newaxis, :], (1, args.timesteps, 1))
            else:  # Already 3D
                reshaped_inputs = np.tile(x, (1, args.timesteps, 1))
            
            y_preds = model.predict(reshaped_inputs, batch_size=32)  # Adjust batch_size for your hardware
            
            # Post-process predictions and actual values
            if args.transformFlag:
                predicted_values = np.expm1(y_preds[:, 0])  # Transform predicted values
                
            else:
                predicted_values = y_preds[:, 0]  # Directly take predictions
                
            
            # Reshape to required dimensions
            predicted_values = predicted_values.reshape(-1, 1)
            
            
            # with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
            #     data = pickle.load(file)
            # data['predicted_weight_RNN'] = predicted_values
            # labels = assign_labels(list(data['data_contextual_weight'].index), boundaries)
            # for p in np.unique(labels):
            #     data[p-1]['df']['predicted_weight_RNN']= predicted_values[labels==p]
            # with open(args.path  + '/dynamic_individual_weight.pkl', 'wb') as file:
            #     pickle.dump(data, file)
            
 

            

if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
    for i in range(len(data)-1):
        args.time_windows_size.append(len(data[i]['df']))
        args.time_windows.append(" (From: "+ str(data[i]['start_date'])+ " - To: "+ str(data[i]['end_date']) + ")\n Sample per day: "+str(data[i]["sampling_rate_per_day"]))
    data_all = data['data_contextual_weight']
    data_all = data[args.selected_time_windows]['df']
    data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp"],axis=1)
        

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
    
    run_name = ""
    if args.reducedFeature:
        run_name += "ReducedFeature_"
        
    if args.transformFlag:
        run_name += "WithTransform_"
    else:
        run_name += "WithoutTransform_"
        
    if args.withTime:
        run_name += "WithTime_"
    else:
        run_name += "WithoutTime_"
        
    if args.scale_flag:
        run_name += "WithScaling_"
    else:
        run_name += "WithoutScaling_"
        
    if args.data_augmentation:
        run_name += "data_augmentation_"
    else:
        run_name += "No_data_augmentation_"
        
    run_name += f"({formatted_datetime})"


    args.run_name = str(args.timesteps)+"_"+args.prediction_Method + "_" + run_name
    args.path = f"data/Runs_dataAugmentation/{args.run_name}"
    args.model_file = args.path + '/fish_weight_prediction_model.keras'
    train(args,data_all)





   
        


