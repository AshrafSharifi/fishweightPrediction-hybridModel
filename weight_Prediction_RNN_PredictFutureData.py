from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
import numpy as np
import pickle
import tyro
from Custom_plots import Custom_plots 
from ModelClass import ModelClass 
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from general import general


@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="CNN_LSTM" 
    
    if prediction_Method!="Random_Forest":
        verbos= 0
        epochs: int= 150
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
    else:
        max_depth: int= 10
        min_samples_leaf: int= 5
        min_samples_split: int= 5
        n_estimators: int= 1000
        random_state: int= 23
        scale_flag: bool() = False 
        transformFlag: bool() = False 
    
    
    using_saved_test_set = True   
    save_test_set = False
    futer_subset_size = 20
    subset_size: float = 100
    withTime: bool() = True
    reducedFeature: bool() = False    
    root = 'data/Preore_Dataset/'
    path=""
    model_file = ""
    displayCorrMatrix = True
    feature_names = []
    # Scalers
    scaler_x = MinMaxScaler()
    time_windows_size = []
    time_windows = []
    run_name=""
    run_folder="Runs_FuturePredict"

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

    # Drop the original timestamp column
    data = data.drop(["Entrance_timestamp"], axis=1)

    # Identify columns to remove for reduced feature dataset
    reduced_columns_to_remove = []
    if args.reducedFeature:
        reduced_columns_to_remove = [
            "Energy_Acquisition(A)", "Catabolic_component(C)", "Somatic_tissue_energy_content(Epsilon)","I_Ration_Per_SamplingFrequency","Feed_ration"
        ]
        

    # Prepare target (y) and features (x)
    y = data["PREORE_VAKI-Weight [g]"].values
    x = data.drop(["PREORE_VAKI-Weight [g]", "mathematical_computed_weight"], axis=1)

    # Create reduced feature dataset
    reduced_x = x.drop(columns=reduced_columns_to_remove, errors='ignore')

    # Save feature names
    args.feature_names = x.columns.tolist()
    args.reduced_feature_names = reduced_x.columns.tolist()

    # Apply transformation to target if flag is set
    if args.transformFlag:
        y = np.log1p(y)

    # Scale datasets if needed
    if args.scale_flag:
        x = args.scaler_x.fit_transform(x)
        reduced_x = args.scaler_x.fit_transform(reduced_x)
    else:
        x = x.values
        reduced_x = reduced_x.values

    return np.array(x), np.array(reduced_x), y, data

def split_data(data, subset, boundaries, timestamp_field=None):
    """
    Splits the data into training and testing sets for each time window
    based on the specified subset percentage. Keeps `subset` percent of data
    from each window as training and the rest as testing.

    Args:
    - data: The dataset to split (can be a numpy array).
    - subset: Percentage of each time window to keep as training data.
    - boundaries: List of tuples defining the start and end indices of each time window.
    - timestamp_field: Not used for numpy arrays but kept for compatibility.

    Returns:
    - train_indices: Indices of the training set.
    - test_indices: Indices of the testing set.
    """
    subset_fraction = subset / 100
    
    if subset_fraction < 0 or subset_fraction > 1:
        raise ValueError("Subset percentage must be between 0 and 100.")
    
    train_indices = []
    test_indices = []
    
    for start, end in boundaries:
        # Extract the range of indices for the current window
        indices = list(range(start, end + 1))
        
        # Calculate the split point
        split_point = int(len(indices) * subset_fraction)
        
        # Determine train and test indices
        keep_indices = indices[:split_point]
        remove_indices = indices[split_point:]
        
        futer_subset_size = args.futer_subset_size / 100
        if subset !=  100:
            # keep futer_subset_size% of data for validation
            split_point = int(len(remove_indices) * futer_subset_size)
            remove_indices = remove_indices[:split_point]
        else:
            all_indices = np.arange(start, end)
            # Calculate the number of indices to select (20%)
            num_to_select = int(len(all_indices) * futer_subset_size)
            # Randomly select indices
            remove_indices = np.random.choice(all_indices, size=num_to_select, replace=False)
            
        
        train_indices.extend(keep_indices)
        test_indices.extend(remove_indices)
    
    return train_indices, test_indices




def train(args,original_data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # Convert time window sizes into ranges
    ranges = [0] + args.time_windows_size  # Add a starting point at 0
    ranges = np.cumsum(ranges)
    boundaries = [(ranges[i], ranges[i+1]-1) for i in range(len(args.time_windows_size))]
    total_metrics = dict()
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in args.feature_names])
    writer.add_text("3: Feature Names", feature_table)
    data_all = original_data.copy()
    x, _, y, data_contained_fishWeight = prepare_data(args, data_all)
    
    for i, subset in enumerate([40,60,80,100]):
        args.subset_size = subset

       

        fold_metrics = {"mse": [],"mae": [],"mape": []}

        train_index, val_index = split_data(x,subset,boundaries)        
 
        x_train, y_train = x[train_index], y[train_index]                                                           
        x_val, y_val = x[val_index],y[val_index]

        # Step 2: Reshape to sequences after train-test split
        samples_train = int(x_train.shape[0] / args.timesteps)
        x_train = x_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps, x_train.shape[1])
        y_train = y_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps)


        original_y_test = y_val
        samples_test = int(x_val.shape[0] / args.timesteps)
        y_val = y_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps)
        original_x_test = x_val
   
        # Step 2: Reshape to sequences after train-test split
        x_val = x_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps, x_val.shape[1])

        labels = general.assign_labels(val_index, boundaries)
        train_labels = general.assign_labels(train_index, boundaries)
        # Fish_Weight_Predictedby_Math_model = np.array(data_contained_fishWeight.iloc[val_index]["Fish_Weight"]).reshape(-1,1)

        modelClass = ModelClass(args.timesteps, x_train.shape[2], args.dropout, args.learning_rate)
        x_train_temp = x_train

            
        if args.prediction_Method=="LSTM":
            model = modelClass.create_LSTM()
        elif args.prediction_Method=="LSTM_CNN":
            model = modelClass.create_lstm_cnn_model()
        elif args.prediction_Method=="CNN_LSTM":
            model = modelClass.create_cnn_lstm_model()
        else:
            model = modelClass.create_parallel_cnn_lstm_model()
        
        # model = modelClass.create_LSTM()
         
        args.model_file =args.path +'/'+str(subset)+ '_'+ 'fish_weight_prediction_model.hdf5' 
        checkpointer = ModelCheckpoint(filepath=args.model_file, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
        
        model.fit(
            x_train_temp, y_train, epochs=args.epochs, batch_size=args.batch_size,
            validation_split=args.validation_split, callbacks=[early_stopping,checkpointer], verbose=args.verbos)
        

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
        
        
           
        for p in np.unique(labels):
            title = "Time window: " +str(p)+ args.time_windows[p-1] + "  (Train size: "+str(len(train_labels[train_labels==p]))+")"
            plots = Custom_plots(predicted_values[labels==p],actual_values[labels==p],writer=writer,title=title,summarytitle=args.run_name+"(TimeWindow: "+str(p)+")"+"(Subset: "+str(subset)+")")
            plots.plot_all()
           
        total_metrics[str(subset)+str(args.reducedFeature)] = {"train_size": int(len(x_train)),"val_size": int(len(x_val)),"subset": subset, "mse": mse1, "mae": mae1, "mape": mape1}

            
    print("*****************************************************************************************")  
    # Initialize the table header
    table_header = "| Metric |"
    table_rows = {"MSE": "| MSE   |", "MAE": "| MAE   |", "MAPE": "| MAPE  |","Train size": "| Train_size  |","Test size": "| Test_size  |"}
    
    # Build the table dynamically
    for _, metrics in total_metrics.items():
        subset =  metrics["subset"]
        table_header += f" {args.prediction_Method}_Training with first{subset}% |"
        # Add the metrics to their respective rows
        table_rows["MSE"] += f" {metrics['mse']:.4f} |"
        table_rows["MAE"] += f" {metrics['mae']:.4f} |"
        table_rows["MAPE"] += f" {metrics['mape']:.4f} |"
        table_rows["Train size"] += f" {metrics['train_size']:.4f} |"
        table_rows["Test size"] += f" {metrics['val_size']:.4f} |"
    
    # Combine the header and rows into a complete table
    table = f"{table_header}\n|-{''.join(['|-'] * len(total_metrics))}-|\n"
    table += "\n".join(table_rows.values())
    
    # Save or display the table
    writer.add_text("1: Metrics Comparison", table)
    
        # Save the table as text (or log it to a writer, etc.)
    writer.add_text(f"Metrics for Subset: {subset}", table)

    writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
    for i in range(len(data)-1):
        args.time_windows_size.append(len(data[i]['df']))
        args.time_windows.append(" (From: "+ str(data[i]['start_date'])+ " - To: "+ str(data[i]['end_date']) + ")\n Sample per day: "+str(data[i]["sampling_rate_per_day"]))
    data_all = data['data_contextual_weight']
    data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp"],axis=1)
        
    for i in range(0,1):
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
            
        run_name += f"({formatted_datetime})"

        args.run_name = str(args.timesteps)+"_"+args.prediction_Method + "_" + run_name
        args.path = f"data/Runs_FuturePredict/{args.run_name}"
        args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
        train(args,data_all)

    



    
   
            
    
    
