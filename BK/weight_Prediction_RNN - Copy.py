from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # high-level visualization based on matplotlib
import pickle
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Custom_plots import *
from ModelClass import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    # 3_LSTM_CNN_WithoutTransform_WithTime_(2024-11-06_11_14_40)
    prediction_Method:str ="LSTM_CNN" 
    
    if prediction_Method!="Random_Forest":
        verbos= 0
        epochs: int= 200
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
    
    
    withTime: bool() = True
    reducedFeature: bool() = False    
    root = 'data/Preore_Dataset/'
    path=""
    model_file = ""
    displayCorrMatrix = True
    feature_names = []
    # Scalers
    scaler_x = MinMaxScaler()

    
    

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
        
def prepare_data(args,data):
    data = data.drop(data.columns[data.columns.str.contains('EXIT')], axis=1)
    data['Entrance_timestamp'] = pd.to_datetime(data['Entrance_timestamp'])
    data = data.drop(["PREORE_VAKI-Length [mm]"], axis=1)
    data = data.drop(["PREORE_VAKI-CF"], axis=1)
    
    Entrance_timestamp = data['Entrance_timestamp'].copy()
    if args.withTime:
        data['year'] = data['Entrance_timestamp'].dt.year
        data['month'] = data['Entrance_timestamp'].dt.month
        data['day_of_week'] = data['Entrance_timestamp'].dt.day_of_week
        data['day_of_month'] = data['Entrance_timestamp'].dt.days_in_month
        data['hour'] = data['Entrance_timestamp'].dt.hour
    data = data.drop(["Entrance_timestamp"], axis=1)

    if args.reducedFeature:
        data = data.drop(["Energy_Acquisition(A)"], axis=1)
        data = data.drop(["Catabolic_component(C)"], axis=1)
        data = data.drop(["Somatic_tissue_energy_content(Epsilon)"], axis=1)
        # data = data.drop(["PREORE_VAKI-Length [mm]"], axis=1)
    
    data_out = data.copy() # data contained Fish_weight
    y = data["PREORE_VAKI-Weight [g]"].values
    x = data.drop(columns=["PREORE_VAKI-Weight [g]"])
    x = data.drop(columns=["Fish_Weight"])
    
    
    if args.transformFlag:
        y = np.log1p(y)  # Use log1p for stability        

    if args.scale_flag:
        x = args.scaler_x.fit_transform(x)
    
  

    return x, y, data

def split_data(x,y,data_contained_fishWeight):
    # Step 1: Train-Test Split
    indices = np.arange(len(x))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=0.2, random_state=23)
    
    original_x_test = x_test.copy()
    original_y_test = y_test.copy()
    
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

def train_random_forest(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "3: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    x, y, data, fishweight_col_indx = prepare_data(args, data)
    x = np.array(x)
    y = np.array(y)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    
    x_test_contained_fishWeight = x_test.copy()
    x_test = np.delete(x_test, fishweight_col_indx, axis=1)
    x_train = np.delete(x_train, fishweight_col_indx, axis=1)
    data = data.drop(columns=["Fish_Weight"])
    # best_rf = find_best_RF_parameters(x_train,y_train)
   
    best_rf = RandomForestRegressor(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    best_rf.fit(x_train, y_train.ravel())
    y_pred = best_rf.predict(x_test)
    actual_value = y_test
    Fish_Weight_Predictedby_Math_model = x_test_contained_fishWeight[:,fishweight_col_indx]
    Fish_Weight_Predictedby_Math_model = Fish_Weight_Predictedby_Math_model.reshape(-1, 1)
    if args.transformFlag and args.scale_flag:
        y_pred = np.expm1(args.scaler_y.inverse_transform(y_pred.reshape(-1, 1)))
        actual_value = np.expm1(args.scaler_y.inverse_transform(y_test))
        Fish_Weight_Predictedby_Math_model = args.scaler_x.inverse_transform(x_test_contained_fishWeight)[:,fishweight_col_indx]
        Fish_Weight_Predictedby_Math_model = Fish_Weight_Predictedby_Math_model.reshape(-1, 1)
    elif args.transformFlag==False and args.scale_flag:
        y_pred = args.scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        actual_value = args.scaler_y.inverse_transform(y_test)
        Fish_Weight_Predictedby_Math_model = args.scaler_x.inverse_transform(x_test_contained_fishWeight)[:,fishweight_col_indx]
        
    elif args.transformFlag==True and args.scale_flag==False:
        y_pred = np.expm1(y_pred.reshape(-1, 1))
        actual_value = np.expm1(y_test).reshape(-1, 1)
        
    
            
            
    y_pred = y_pred.reshape(-1, 1) 
    Fish_Weight_Predictedby_Math_model = Fish_Weight_Predictedby_Math_model.reshape(-1, 1)
    actual_value = actual_value.reshape(-1, 1) 
    mse1,mae1,mape1= compute_metrics(np.array(y_pred), np.array(actual_value))
    

    # Compute and log evaluation metrics
    mse = mean_squared_error(actual_value, y_pred)
    mae = mean_absolute_error(actual_value, y_pred)
    mape = mean_absolute_percentage_error(actual_value, y_pred) * 100
    print(f"Random Forest MSE: {mse}")
    print(f"Random Forest MAE: {mae}")
    print(f"Random Forest MAPE: {mape}%")
    
   
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in data.columns])
    writer.add_text("4: Feature Names", feature_table)
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
    plots = Custom_plots(y_pred, actual_value,writer)
    plots.plot_all()

    
    
    # Add data related to method which is based on mathematical model

    
    mse2,mae2,mape2= compute_metrics(np.array(Fish_Weight_Predictedby_Math_model), np.array(actual_value))
    table_header = f"| Metric | {args.prediction_Method} | Method based on mathematical model |\n|-|-|-|"
    table_rows = f"| MSE   | {mse:.4f} | {mse2:.4f} |\n"
    table_rows += f"| MAE   | {mae:.4f} | {mae2:.4f} |\n"
    table_rows += f"| MAPE  | {mape:.4f} | {mape2:.4f} |"
    table = f"{table_header}\n{table_rows}"
    writer.add_text("1: Metrics Comparison", table)
    
    plots = Custom_plots(Fish_Weight_Predictedby_Math_model, np.array(actual_value),writer)
    plots.plot_all()
    writer.close()




def train_fold(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)
    x, y, data_contained_fishWeight = prepare_data(args, data)
    samples = int(x.shape[0] / args.timesteps)
    x = np.array(x)
    y = np.array(y)
    x = x[:samples * args.timesteps].reshape(samples, args.timesteps, x.shape[1])
    y = y[:samples * args.timesteps].reshape(samples, args.timesteps)
    
    # Initialize KFold
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=23)
    
    fold_metrics = {"mse": [],"mae": [],"mape": []}
   
    for fold, (train_index, val_index) in enumerate(kfold.split(x)):
        print(f"Training fold {fold + 1}/{args.n_splits}")
    
        # Split data into train and validation based on KFold
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Remove fishweight column if required
        x_val_contained_fishWeight = data_contained_fishWeight[:,val_index]
        
        
        # Initialize and compile model
        modelClass = ModelClass(args.timesteps, x.shape[2]-1, args.dropout, args.learning_rate)
        
        if args.prediction_Method=="LSTM":
            model = modelClass.create_LSTM()
        elif args.prediction_Method=="LSTM_CNN":
            model = modelClass.create_lstm_cnn_model()
        elif args.prediction_Method=="CNN_LSTM":
            model = modelClass.create_cnn_lstm_model()
        else:
            model = modelClass.create_parallel_cnn_lstm_model()
        
        # Define callbacks
        # checkpointer = ModelCheckpoint(filepath=f"{args.model_file}_fold{fold+1}", save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
        
        # Train model on this fold
        history = model.fit(
            x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
            validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=args.verbose
        )
        
        # Evaluate the model on validation set for this fold
        val_loss, val_mse, val_mae, val_mape = model.evaluate(x_val, y_val, verbose=0)

        # Predictions vs Actual values plot
        original_x_test = x_val.reshape(-1, x_val.shape[2])
        original_y_test = y_val.reshape(-1)
        
        x_reshaped = x_val_contained_fishWeight.reshape(-1, x_val_contained_fishWeight.shape[2])
        Fish_Weight_Predictedby_Math_model = args.scaler_x.inverse_transform(x_reshaped)[:,fishweight_col_indx]
        
        predicted_values = []
        actual_values = []
        for j in range(len(original_y_test)):
            sample_reshaped = np.tile(original_x_test[j], (1, args.timesteps, 1))
            y_pred = model.predict(sample_reshaped)
            
            if args.transformFlag:
                y_pred = np.expm1(args.scaler_y.inverse_transform(y_pred))[0][0]
                actual_value = np.expm1(args.scaler_y.inverse_transform([[original_y_test[j]]]))[0][0]
            else:
                y_pred = args.scaler_y.inverse_transform(y_pred)[0][0]
                actual_value = args.scaler_y.inverse_transform([[original_y_test[j]]])[0][0]
            
            predicted_values.append(y_pred)
            actual_values.append(actual_value)
        mse1,mae1,mape1= compute_metrics(np.array(predicted_values), np.array(actual_values))
        fold_metrics["mse"].append(mse1)
        fold_metrics["mae"].append(mae1)
        fold_metrics["mape"].append(mape1)
        # plots = Custom_plots(np.array(predicted_values), np.array(actual_values),writer)
        # plots.plot_all()
    
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in data.columns])
    writer.add_text("3: Feature Names", feature_table)


    

    print("*****************************************************************************************")    
    mse2,mae2,mape2= compute_metrics(Fish_Weight_Predictedby_Math_model, np.array(actual_values))
    mse1= np.mean(fold_metrics["mse"])
    mae1= np.mean(fold_metrics["mae"])
    mape1= np.mean(fold_metrics["mape"])
    table_header = f"| Metric | {args.prediction_Method} | Method based on mathematical model |\n|-|-|-|"
    table_rows = f"| MSE   | {mse1:.4f} | {mse2:.4f} |\n"
    table_rows += f"| MAE   | {mae1:.4f} | {mae2:.4f} |\n"
    table_rows += f"| MAPE  | {mape1:.4f} | {mape2:.4f} |"
    table = f"{table_header}\n{table_rows}"
    writer.add_text("1: Metrics Comparison", table)
    
   
    writer.close()


def train(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "2: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in data.columns])
    writer.add_text("3: Feature Names", feature_table)
    
    x, y, data_contained_fishWeight = prepare_data(args, data)
    x_train, x_test, y_train, y_test,data,Fish_Weight_Predictedby_Math_model, original_x_test, original_y_test= split_data( x, y, data_contained_fishWeight)
                                                                        
    modelClass = ModelClass(args.timesteps, x_train.shape[2], args.dropout, args.learning_rate)
    
    model = modelClass.create_LSTM()
    checkpointer = ModelCheckpoint(filepath=args.model_file, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)

    history = model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
        validation_split=args.validation_split, callbacks=[checkpointer, early_stopping], verbose=args.verbos
    )
    
    log_metrics(writer, history)
    for metric in ['loss', 'mse', 'mae', 'mape']:
        fig, ax = plt.subplots()
        ax.plot(history.history[metric], label=f'Training {metric.upper()}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.upper())
        ax.legend()
        writer.add_figure(f"Training/{metric.upper()}", fig)
        plt.close(fig)  
    
    # Final evaluation and predictions
    test_results = model.evaluate(x_test, y_test)


    predicted_values = []
    actual_values = []
    for j in range(len(original_y_test)):
        sample_reshaped = np.tile(original_x_test[j], (1, args.timesteps, 1))
        y_pred = model.predict(sample_reshaped)
        if args.transformFlag:
            y_pred = np.expm1(y_pred)[0][0]
            actual_value = np.expm1([[original_y_test[j]]])[0][0]
        else:
            y_pred = y_pred[0][0]
            actual_value = original_y_test[j]
        
        predicted_values.append(y_pred)
        actual_values.append(actual_value)
    mse1,mae1,mape1= compute_metrics(np.array(predicted_values), np.array(actual_values))
    
    plots = Custom_plots(np.array(predicted_values), np.array(actual_values),writer)
    plots.plot_all()
    plt.close(fig)
    
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
    print("*****************************************************************************************")  
    Fish_Weight_Predictedby_Math_model= np.array(Fish_Weight_Predictedby_Math_model).reshape(-1,1)
    actual_values = np.array(actual_values).reshape(-1,1)
    mse2,mae2,mape2= compute_metrics(Fish_Weight_Predictedby_Math_model, np.array(actual_values))
    table_header = f"| Metric | {args.prediction_Method} | Method based on mathematical model |\n|-|-|-|"
    table_rows = f"| MSE   | {mse1:.4f} | {mse2:.4f} |\n"
    table_rows += f"| MAE   | {mae1:.4f} | {mae2:.4f} |\n"
    table_rows += f"| MAPE  | {mape1:.4f} | {mape2:.4f} |"
    table = f"{table_header}\n{table_rows}"
    writer.add_text("1: Metrics Comparison", table)
    plots = Custom_plots(np.array(Fish_Weight_Predictedby_Math_model), np.array(actual_values),writer,"Method based on mathematical model_")
    plots.plot_all()
    plt.close(fig)
    writer.close()
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
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

        if args.prediction_Method=="Random_Forest":
            run_name = "RF_" + run_name
            args.path = f"data/Runs/{run_name}"
            train_random_forest(args, data_all)
        else:
            run_name = str(args.timesteps)+"_"+args.prediction_Method + "_" + run_name
            args.path = f"data/Runs/{run_name}"
            args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
            train(args,data_all)

    
    # for i in range(0,len(data)-1):
    #     print("RNN on data of each time window "+str(i+1)+"___________________________________")
    #     data_all = data[i]['data_contextual_weight']
    #     data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp","Fish_Weight"],axis=1)
    #     rnn = RNN(data_all)
    #     rnn.train()
    


    
   
            
    
    
