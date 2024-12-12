from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Custom_plots import Custom_plots
from ModelClass import ModelClass


from sklearn.model_selection import KFold

from general import general

@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="LSTM" 
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
    num_splits:int = 10


            
    
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
        data = data.drop(["Energy_Acquisition(A)", "Catabolic_component(C)", "Somatic_tissue_energy_content(Epsilon)","I_Ration_Per_SamplingFrequency","Feed_ration"], axis=1)
    

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
def get_split_indices(args):
    
    
    # Initialize start index for the first list
    start_index = 0
    
    # Store the partitions' start and end indices
    partitions = []
    output = dict()
    tw=0
    # Iterate through each list and partition them into 5 parts
    for size in args.time_windows_size:
        # Calculate the partition size
        partition_size = size // args.num_splits
        remainder = size % args.num_splits  # Remainder to distribute among partitions
    
        # Partition indices for the current list
        current_partitions = []
        current_start = start_index
    
        for i in range(args.num_splits):
            # Add the remainder to the first few partitions
            extra = 1 if i < remainder else 0
            current_end = current_start + partition_size + extra - 1
    
            # Append the partition indices (start, end)
            current_partitions.append((current_start, current_end))
    
            # Update the start index for the next partition
            current_start = current_end + 1
    
        # Add current partitions to the main list
        partitions.append(current_partitions)
    
        # Update start_index for the next list
        start_index += size
        output[tw]=current_partitions
        tw += 1
    
    return output

def get_data_portion(args,x,split_indices,i,boundaries):
    
    train_indices=[]
    val_indices=[]
    tw_labels_val = dict()
    tw_labels_train = dict()
    for tw in range(len(args.time_windows)):
        start_idx, end_idx = split_indices[tw][i][0], split_indices[tw][i][1]
        train_indices_train =list(range(start_idx,end_idx+1))
        train_indices.extend(train_indices_train)
        tw_labels_train[tw]=train_indices_train
        
        if i!=args.num_splits-1:
            start_idx, end_idx = split_indices[tw][i+1][0], split_indices[tw][i+1][1]
            val_indices_set = list(range(start_idx+1,end_idx+1))
            val_indices.extend(val_indices_set)
            tw_labels_val[tw]=val_indices_set
            # print(general.assign_labels(tw_labels_val[tw], boundaries))
        else:

            all_indices = np.arange(split_indices[tw][0][0], split_indices[tw][-1][1])
            # Calculate the number of indices to select (20%)
            num_to_select = int(len(all_indices) * 0.2)
            # Randomly select indices
            val_indices_set = np.random.choice(all_indices, size=num_to_select, replace=False)
            val_indices.extend(val_indices_set)
            tw_labels_val[tw]=val_indices_set
 
    return train_indices, val_indices,tw_labels_train,tw_labels_val
def incremental_train(args, data):
    
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    ranges = [0] + args.time_windows_size  # Add a starting point at 0
    ranges = np.cumsum(ranges)
    boundaries = [(ranges[i], ranges[i+1]-1) for i in range(len(args.time_windows_size))]
    
    x, y, data_contained_fishWeight = prepare_data(args, data)
    
    # Split the dataset into incremental chunks
    # split_indices = []
    # for i in range(len(args.time_windows)):
    #     split_indices.append(np.linspace(0, args.time_windows_size[i], args.num_splits, dtype=int))
    
    split_indices= get_split_indices(args)

    fold_metrics = {"mse": [], "mae": [], "mape": []}
    incremental_model = None
    total_metrics = dict()
    for i in range(args.num_splits):
        print(f"Incremental Training Step {i+1}/{args.num_splits}")
        
       
        train_indices, val_indices, tw_labels_train,tw_labels_val = get_data_portion(args,x,split_indices,i,boundaries)
        # Get current training data chunk
        x_train_chunk, y_train_chunk = x[train_indices], y[train_indices]
        
        # Reshape for LSTM input
        samples_chunk = x_train_chunk.shape[0] // args.timesteps
        x_train_chunk = x_train_chunk[:samples_chunk * args.timesteps].reshape(samples_chunk, args.timesteps, x_train_chunk.shape[1])
        y_train_chunk = y_train_chunk[:samples_chunk * args.timesteps].reshape(samples_chunk, args.timesteps)
        
        
        # Get current validation data chunk
        
        x_val_chunk, y_val_chunk = x[val_indices], y[val_indices]
        original_x_test = x_val_chunk
        original_y_test = y_val_chunk
        # Reshape for LSTM input
        samples_chunk = x_val_chunk.shape[0] // args.timesteps
        x_val_chunk = x_val_chunk[:samples_chunk * args.timesteps].reshape(samples_chunk, args.timesteps, x_val_chunk.shape[1])
        y_val_chunk = y_val_chunk[:samples_chunk * args.timesteps].reshape(samples_chunk, args.timesteps)
        
        
        # Initialize or reload the model
        if incremental_model is None:
            
            # Create Model
            modelClass = ModelClass(args.timesteps, x_train_chunk.shape[2], args.dropout, args.learning_rate)
            if args.prediction_Method == "LSTM":
                model = modelClass.create_LSTM()
            elif args.prediction_Method == "LSTM_CNN":
                model = modelClass.create_lstm_cnn_model()
            elif args.prediction_Method == "CNN_LSTM":
                model = modelClass.create_cnn_lstm_model()
            else:
                model = modelClass.create_parallel_cnn_lstm_model()
            incremental_model = model
        else:
            print("Loading previously trained model weights...")
            incremental_model.load_weights(args.model_file)
        
        # Define callbacks
        checkpointer = ModelCheckpoint(filepath=args.model_file, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)

        # Train the model on the current chunk
        history = incremental_model.fit(
            x_train_chunk, y_train_chunk, 
            epochs=args.epochs, batch_size=args.batch_size,
            validation_split=args.validation_split, 
            callbacks=[early_stopping, checkpointer], verbose=args.verbose
        )
        
        general.log_metrics(writer, history)
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
        
        
        for p in range(len(args.time_windows)):
            title = "Time window: " +str(p+1)+ args.time_windows[p] + "  (Train size: "+str(len(tw_labels_train[p]))+")"
            val_label = general.assign_labels(val_indices, boundaries)
            plots = Custom_plots(predicted_values[val_label==p+1],actual_values[val_label==p+1],writer=writer,title=title,summarytitle=args.run_name+f"_Incremental Training Step {i+1} (TimeWindow:{p+1})")
            plots.plot_all()
            plt.close(fig)
        print("*****************************************************************************************")  
        
        total_metrics[str(i)] = {"train_size": int(len(train_indices)),"val_size": int(len(val_indices)), "mse": mse1, "mae": mae1, "mape": mape1}


    print("*****************************************************************************************")  
    # Initialize the table header
    table_header = "| Metric |"
    table_rows = {"MSE": "| MSE   |", "MAE": "| MAE   |", "MAPE": "| MAPE  |","Train size": "| Train_size  |","Test size": "| Test_size  |"}
    
    # Build the table dynamically
    for key, metrics in total_metrics.items():
       
        table_header += f" Step{key} |"
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
    writer.add_text(f"Metrics for Increamental Learning", table)

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
        args.path = f"data/Runs_IncreamentalLearning/{args.run_name}"
        args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
        incremental_train(args,data_all)

    


    
   
            
    
    
