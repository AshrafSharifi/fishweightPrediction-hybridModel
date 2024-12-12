from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tyro
from Custom_plots import Custom_plots 
from ModelClass import ModelClass 
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import shap


@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="LSTM" 
    
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
    run_folder="Runs_SubsetData"


def find_best_RF_parameters(x_train,y_train):
    #Define parameter grid for GridSearch
    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [5, 6, 7, 8, 9, 10],
        "min_samples_split": [5, 10, 25, 50],
        "min_samples_leaf": [5, 10, 25, 50]
    }
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(random_state=23)
    # Initialize Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=args.verbos)
    # Fit the model with GridSearch
    grid_search.fit(x_train, y_train.ravel())
    # Retrieve the best model
    best_rf = grid_search.best_estimator_
    print("Best Random Forest Parameters:", grid_search.best_params_)
    return best_rf

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






def compute_metrics(predicted_values,actual_values):
    # Calculate loss, MSE, MAE, and MAPE
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    return mse,mae,mape

    
  
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
    
def shap_feature_selection(args, best_rf, x_train, writer):
    # Initialize SHAP TreeExplainer
    shap_explainer = shap.TreeExplainer(best_rf)
    shap_importance_train = shap_explainer.shap_values(x_train)

    # Save the summary plot (beeswarm) as an image and add it to TensorBoard
    fig = plt.figure()
    shap.summary_plot(shap_importance_train, x_train, feature_names=args.feature_names, show=False)
    writer.add_figure("SHAP Summary Plot (Beeswarm)", fig)
    

    # Save the bar plot summary as an image and add it to TensorBoard
    fig = plt.figure()
    shap.summary_plot(shap_importance_train, x_train, feature_names=args.feature_names, plot_type="bar", show=False)
    writer.add_figure("SHAP Summary Plot (Bar)", fig)


def assign_labels(val_index, boundaries):
    labels = np.zeros_like(val_index, dtype=int)
    for i, (start, end) in enumerate(boundaries):
        labels[(val_index >= start) & (val_index <= end)] = i + 1
    return labels

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
    
    for i, subset in enumerate([100, 100, 90, 80, 70, 60, 50]):
        args.subset_size = subset
        if args.subset_size==100:
            data_all = original_data['data_contextual_weight']
            if i==0:
                args.reducedFeature = True
                args.save_test_set = True
                args.using_saved_test_set = False
            else:
                args.reducedFeature = False
                args.save_test_set = False
                args.using_saved_test_set = True
        else:
            data_all = remove_random_records(args,original_data,'Entrance_timestamp')
       
        data = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp"],axis=1)
        x, reduced_x, y, data_contained_fishWeight = prepare_data(args, data)
        data = data_contained_fishWeight.drop(columns=["mathematical_computed_weight"])
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=23)
        fold_metrics = {"mse": [],"mae": [],"mape": []}
        
        testset_dict = dict()
        if args.using_saved_test_set:
            with open('data/testset.pkl', 'rb') as file:
                testset_dict = pickle.load(file)
                
        for fold, (train_index, val_index) in enumerate(kfold.split(x)):
            print(f"Training fold {fold + 1}/{args.n_splits}")     
            x_train, y_train = x[train_index], y[train_index]                                                           
            x_val, y_val = x[val_index],y[val_index]
            reduced_x_train, reduced_x_val= reduced_x[train_index],reduced_x[val_index]
            
            # Step 2: Reshape to sequences after train-test split
            samples_train = int(x_train.shape[0] / args.timesteps)
            x_train = x_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps, x_train.shape[1])
            reduced_x_train = reduced_x_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps, reduced_x_train.shape[1])
            y_train = y_train[:samples_train * args.timesteps].reshape(samples_train, args.timesteps)
    
            
            
            if args.using_saved_test_set:
                original_reduced_x_test = testset_dict[str(fold)]["original_reduced_x_test"]
                reduced_x_val = testset_dict[str(fold)]["reduced_x_val"] 
                original_x_test = testset_dict[str(fold)]["original_x_test"]
                x_val = testset_dict[str(fold)]["x_val"]
                original_y_test = testset_dict[str(fold)]["original_y_test"]
                y_val = testset_dict[str(fold)]["y_val"]
                val_index = testset_dict[str(fold)]["val_index"]
            else:
                original_y_test = y_val
                samples_test = int(x_val.shape[0] / args.timesteps)
                y_val = y_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps)
                original_x_test = x_val
                original_reduced_x_test = reduced_x_val
                # Step 2: Reshape to sequences after train-test split
                x_val = x_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps, x_val.shape[1])
                reduced_x_val = reduced_x_val[:samples_test * args.timesteps].reshape(samples_test, args.timesteps, reduced_x_val.shape[1])
                if args.save_test_set:
                    testset_dict_item= dict()
                    testset_dict_item["original_x_test"]=original_x_test
                    testset_dict_item["original_reduced_x_test"]=original_reduced_x_test
                    testset_dict_item["original_y_test"]=original_y_test
                    testset_dict_item["y_val"]=y_val
                    testset_dict_item["x_val"]=x_val
                    testset_dict_item["reduced_x_val"]=reduced_x_val 
                    testset_dict_item["val_index"]=val_index
                    testset_dict[str(fold)]=testset_dict_item
                

            labels = assign_labels(val_index, boundaries)
            train_labels = assign_labels(train_index, boundaries)
            # Fish_Weight_Predictedby_Math_model = np.array(data_contained_fishWeight.iloc[val_index]["Fish_Weight"]).reshape(-1,1)
            if args.reducedFeature:
                modelClass = ModelClass(args.timesteps, reduced_x_train.shape[2], args.dropout, args.learning_rate)
                x_train_temp = reduced_x_train
            else:
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
             
        
            early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
            
            history = model.fit(
                x_train_temp, y_train, epochs=args.epochs, batch_size=args.batch_size,
                validation_split=args.validation_split, callbacks=[early_stopping], verbose=args.verbos)
            
            if fold + 1 == args.n_splits:
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
        
            # Ensure inputs are 3D by adding a dimension if necessary
            if args.reducedFeature:
                if original_reduced_x_test.ndim == 2:
                    reshaped_inputs = np.tile(original_reduced_x_test[:, np.newaxis, :], (1, args.timesteps, 1))
                else:  # Already 3D
                    reshaped_inputs = np.tile(original_reduced_x_test, (1, args.timesteps, 1))
            else:
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

            
            mse1,mae1,mape1= compute_metrics(predicted_values, actual_values)
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
                total_metrics[str(subset)+str(args.reducedFeature)] = {"reducedFeature": args.reducedFeature,"subset": subset, "mse": mse1, "mae": mae1, "mape": mape1}
        
        if args.save_test_set:
            with open('data/testset.pkl', 'wb') as file:
                pickle.dump(testset_dict, file)            
    print("*****************************************************************************************")  
    # Initialize the table header
    table_header = "| Metric |"
    table_rows = {"MSE": "| MSE   |", "MAE": "| MAE   |", "MAPE": "| MAPE  |"}
    
    # Build the table dynamically
    for _, metrics in total_metrics.items():
        subset =  metrics["subset"]
        table_header += f" {args.prediction_Method}_Training with {subset}% data_ReducedFeatures:{str(metrics['reducedFeature'])} |"
        # Add the metrics to their respective rows
        table_rows["MSE"] += f" {metrics['mse']:.4f} |"
        table_rows["MAE"] += f" {metrics['mae']:.4f} |"
        table_rows["MAPE"] += f" {metrics['mape']:.4f} |"
    
    # Combine the header and rows into a complete table
    table = f"{table_header}\n|-{''.join(['|-'] * len(total_metrics))}-|\n"
    table += "\n".join(table_rows.values())
    
    # Save or display the table
    writer.add_text("1: Metrics Comparison", table)
    
        # Save the table as text (or log it to a writer, etc.)
    writer.add_text(f"Metrics for Subset: {subset}", table)
    

    
    # for p in np.unique(labels):
    #     title = "Method based on mathematical model\nTime window: " +str(p)+ args.time_windows[p-1]
    #     plots = Custom_plots(Fish_Weight_Predictedby_Math_model[labels==p],actual_values[labels==p],writer=writer, title=title, summarytitle="Method based on mathematical model(TimeWindow: "+str(p)+")")
    #     plots.plot_all()
    #     plt.close(fig)
    # plots = Custom_plots(Fish_Weight_Predictedby_Math_model, actual_values,writer,"Method based on mathematical model_")
    # plots.plot_all()
    # plt.close(fig)
    writer.close()



def remove_last_records(args,data, timestamp_field='Entrance_timestamp'):
    """
    Removes the specified percentage of records dynamically based on time window sizes.

    """
    
    time_window_sizes = args.time_windows_size
    percentage = (100 - args.subset_size)/100
    
    
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1.")
    
    # Total number of records to remove
    total_to_remove = int(len(data['data_contextual_weight']) * percentage)
    
    # Calculate the number of records to remove from each time window
    total_size = sum(time_window_sizes)
    proportions = [size / total_size for size in time_window_sizes]
    to_remove_per_window = [int(total_to_remove * p) for p in proportions]
    
    modified_windows = []
    for i in range(len(data)-1):
        window = data[i]['df']
        window = window.sort_values(by=timestamp_field)
        modified_windows.append(window.iloc[:-to_remove_per_window[i]])
    

    
    # Combine the modified windows back into a single DataFrame
    remaining_data = pd.concat(modified_windows, ignore_index=True)
    return remaining_data



def remove_random_records(args, data, timestamp_field='Entrance_timestamp'):
  
    time_window_sizes = args.time_windows_size
    percentage = (100 - args.subset_size) / 100  # percentage to remove
    
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1.")
    
    # Calculate total number of records to remove
    total_to_remove = int(len(data['data_contextual_weight']) * percentage)
    
    # Calculate the number of records to remove from each time window proportionally
    total_size = sum(time_window_sizes)
    proportions = [size / total_size for size in time_window_sizes]
    to_remove_per_window = [int(total_to_remove * p) for p in proportions]
    
    modified_windows = []
    start_idx = 0  # to keep track of the starting index for each window
    
    for i in range(len(data)-1):
        window = data[i]['df']
        
        # Randomly sample and remove rows from the window
        window_size = len(window)
        num_to_remove = to_remove_per_window[i]
        
        if num_to_remove > 0:
            # Randomly select 'num_to_remove' rows to remove
            rows_to_remove = window.sample(n=num_to_remove, random_state=42)  # Set random_state for reproducibility
            window = window.drop(rows_to_remove.index)  # Drop the randomly selected rows
        
        # Append the modified window to the list
        modified_windows.append(window)
    
    # Combine the modified windows back into a single DataFrame
    remaining_data = pd.concat(modified_windows, ignore_index=True)
    return remaining_data

    

            

if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
    for i in range(len(data)-1):
        args.time_windows_size.append(len(data[i]['df']))
        args.time_windows.append(" (From: "+ str(data[i]['start_date'])+ " - To: "+ str(data[i]['end_date']) + ")\n Sample per day: "+str(data[i]["sampling_rate_per_day"]))
    
   
        
    for i in range(0,1):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
        
        # run_name = str(args.subset_size) + "_"
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
        args.path = f"data/{args.run_folder}/{args.run_name}"
        # args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
        train(args,data)

    



    
   
            
    
    
