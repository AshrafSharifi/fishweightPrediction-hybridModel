from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ModelClass import ModelClass
from general import general
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf


@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="LSTM_CNN" 
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
    withTime: bool() = True
    reducedFeature: bool() = False    
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
    num_splits:int = 1
    selected_tw = 1
    subset_size = 100


            
    
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

    # Drop additional columns based on reduced feature flag
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
    
    x = np.array(x)
    return x, y, data
def get_split_indices(args, data, timestamp_field='Entrance_timestamp'):
    """
    Returns the indices of the first and last elements of the subset of data 
    for each time window based on args.subset_size.
    
    Args:
        args: Object containing time_windows_size (list of time window sizes) 
              and subset_size (percentage of data to retain).
        data: The data containing the time windows (list of DataFrames).
        timestamp_field: The timestamp field used to define the data order.
        
    Returns:
        A dictionary where keys are the time window indices, and values are 
        tuples (first_index, last_index) for the retained subset.
    """
    time_window_sizes = args.time_windows_size
    subset_percentage = args.subset_size / 100  # percentage to retain
    
    if subset_percentage < 0 or subset_percentage > 1:
        raise ValueError("Subset size percentage must be between 0 and 100.")
    
    # Dictionary to store the first and last indices for each time window
    first_last_indices = {}
    
   
        
        
    # Calculate the number of records to retain
    num_to_retain = int(len(data) * subset_percentage)
    
    if num_to_retain > 0:
        # Get the first and last index of the subset
        first_index = data.index[0]
        last_index = data.index[num_to_retain - 1]
        first_last_indices[i] = (first_index, last_index)
    else:
        # If no records are retained, set indices to None
        first_last_indices[i] = (None, None)
    
    return first_last_indices




def get_data_portion(args,x,split_indices,i,boundaries):
    
    train_indices=[]
    val_indices=[]


    start_idx, end_idx = split_indices[args.selected_tw][i][0], split_indices[args.selected_tw][i][1]
    train_indices_train =list(range(start_idx,end_idx+1))
    train_indices, val_indices = train_test_split(train_indices_train, test_size=0.2, random_state=42)

        # if i!=args.num_splits-1:
        #     train_indices_train = np.random.choice(train_indices_train, size=.2, replace=False)
        #     train_indices.extend(train_indices_train)
        #     tw_labels_val[tw]=val_indices_set
        #     # print(general.assign_labels(tw_labels_val[tw], boundaries))
        # else:

        #     all_indices = np.arange(split_indices[tw][0][0], split_indices[tw][-1][1])
        #     # Calculate the number of indices to select (20%)
        #     num_to_select = int(len(all_indices) * 0.2)
        #     # Randomly select indices
        #     val_indices_set = np.random.choice(all_indices, size=num_to_select, replace=False)
        #     val_indices.extend(val_indices_set)
        #     tw_labels_val[tw]=val_indices_set
 
    return train_indices, val_indices


def incremental_train(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    ranges = [0] + args.time_windows_size
    ranges = np.cumsum(ranges)

    split_indices = get_split_indices(args, data)
    x, y, data_contained_fishWeight = prepare_data(args, data)

    total_metrics = dict()
    
    @tf.function
    def predict_batch(model, inputs):
        """Efficient batch prediction with tf.function."""
        return model(inputs, training=False)
    
    for i in range(args.num_splits):
        print(f"Subset {args.subset_size}")
        
        start_idx, end_idx = split_indices[0][0], split_indices[0][1]
        all_indices = list(range(start_idx, end_idx + 1))
        x_all = x[all_indices]
        
        # kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=23)
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics = {"mse": [], "mae": [], "mape": []}
        
        for fold, (train_index, val_index) in enumerate(tscv.split(x_all)):
            print(f"Training fold {fold + 1}/{args.n_splits}")
            
            # Prepare train and validation data
            x_train_chunk, x_val_chunk = x[train_index], x[val_index]
            y_train_chunk, y_val_chunk = y[train_index], y[val_index]
            
            # Reshape for LSTM input
            samples_chunk = x_train_chunk.shape[0] // args.timesteps
            x_train_chunk = x_train_chunk[:samples_chunk * args.timesteps].reshape(
                samples_chunk, args.timesteps, x_train_chunk.shape[1])
            y_train_chunk = y_train_chunk[:samples_chunk * args.timesteps].reshape(
                samples_chunk, args.timesteps)
            
            samples_chunk = x_val_chunk.shape[0] // args.timesteps
            x_val_chunk = x_val_chunk[:samples_chunk * args.timesteps].reshape(
                samples_chunk, args.timesteps, x_val_chunk.shape[1])
            y_val_chunk = y_val_chunk[:samples_chunk * args.timesteps].reshape(
                samples_chunk, args.timesteps)
            
            # Model initialization
            modelClass = ModelClass(args.timesteps, x_train_chunk.shape[2], args.dropout, args.learning_rate)
            if args.prediction_Method == "LSTM":
                model = modelClass.create_LSTM()
            elif args.prediction_Method == "LSTM_CNN":
                model = modelClass.create_lstm_cnn_model()
            elif args.prediction_Method == "CNN_LSTM":
                model = modelClass.create_cnn_lstm_model()
            else:
                model = modelClass.create_parallel_cnn_lstm_model()
            
            # Train the model
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=args.patience, restore_best_weights=True)
            model.fit(
                x_train_chunk, y_train_chunk,
                epochs=args.epochs, batch_size=args.batch_size,
                validation_split=args.validation_split, 
                callbacks=[early_stopping], verbose=args.verbose
            )
            
            # Ensure inputs are 3D for predictions
            if x_val_chunk.ndim == 2:
                reshaped_inputs = np.expand_dims(x_val_chunk, axis=1)
            else:
                reshaped_inputs = x_val_chunk
            
            # Efficient batch prediction
            y_preds = predict_batch(model, reshaped_inputs).numpy()
            
            # Post-process predictions and actual values
            if args.transformFlag:
                predicted_values = np.expm1(y_preds[:, 0])
                actual_values = np.expm1(y_val_chunk.flatten())
            else:
                predicted_values = y_preds[:, 0]
                actual_values = y_val_chunk.flatten()
            
            predicted_values = predicted_values.reshape(-1, 1)
            actual_values = actual_values.reshape(-1, 1)
            
            mse1, mae1, mape1 = general.compute_metrics(predicted_values, actual_values)
            fold_metrics["mse"].append(mse1)
            fold_metrics["mae"].append(mae1)
            fold_metrics["mape"].append(mape1)
            
            del model
            tf.keras.backend.clear_session()
        
        # Aggregate fold metrics
        total_metrics[f"Step_{i + 1}"] = {
            "train_size": len(x_train_chunk),
            "val_size": len(x_val_chunk),
            "mse": np.mean(fold_metrics["mse"]),
            "mae": np.mean(fold_metrics["mae"]),
            "mape": np.mean(fold_metrics["mape"]),
        }
    print(total_metrics)
    # Generate metrics table
    table_header = "| Metric |" + " | ".join(total_metrics.keys()) + " |\n" + "|-" * (len(total_metrics) + 1) + "|\n"
    table_rows = "\n".join(
        f"| {key.upper()} | " + " | ".join(f"{val[key]:.4f}" for val in total_metrics.values()) + " |"
        for key in ["mse", "mae", "mape"]
    )
    table = table_header + table_rows
    writer.add_text("Metrics for Incremental Learning", table)
    writer.close()


    


  
            
            
            
 

            

if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
    for i in range(len(data)-1):
        args.time_windows_size.append(len(data[i]['df']))
        args.time_windows.append(" (From: "+ str(data[i]['start_date'])+ " - To: "+ str(data[i]['end_date']) + ")\n Sample per day: "+str(data[i]["sampling_rate_per_day"]))
    data_all = data[args.selected_tw]['df']
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
        args.path = f"data/Runs_SubsetData/{args.run_name}"
        args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
        incremental_train(args,data_all)

    


    
   
            
    
    
