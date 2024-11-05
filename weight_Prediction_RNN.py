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


@dataclass
class Args:
    # "LSTM" "LSTM_CNN" "CNN_LSTM" "Parrarel_CNN_LSTM" "Random_Forest"
    prediction_Method ="Random_Forest" 
    if prediction_Method!="Random_Forest":
        verbos= 0
        epochs: int= 200
        batch_size: int= 32
        validation_split: float= 0.2
        timesteps: int= 3
        patience: int= 10 
        dropout: float= 0.2 
        learning_rate: float= 0.001  
    else:
        max_depth: int= 10
        min_samples_leaf: int= 5
        min_samples_split: int= 5
        n_estimators: int= 1000
        random_state: int= 23
        
    transformFlag: bool() = True
    scale_flag = True
    withTime: bool() = True
    reducedFeature: bool() = False    
    root = 'data/Preore_Dataset/'
    path=""
    model_file = ""
    displayCorrMatrix = True
    feature_names = []
    # Scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    

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
        
    
    x = data.drop(columns=["PREORE_VAKI-Weight [g]"])
    y = data["PREORE_VAKI-Weight [g]"].values
    
    fish_weight_col_indx = x.columns.get_loc('Fish_Weight')
    if args.transformFlag:
        y = np.log1p(data["PREORE_VAKI-Weight [g]"].values)  # Use log1p for stability        
    # Apply scaling
    x = args.scaler_x.fit_transform(x)
    y = args.scaler_y.fit_transform(y.reshape(-1, 1))
    return x, y, data, fish_weight_col_indx

def compute_metrics(predicted_values,actual_values,writer=None):
    # Calculate loss, MSE, MAE, and MAPE
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    
    if writer != None:
        writer.add_text("2: Evaluation metrics using model.predict on actual data",
        "|Test Metric|Value|\n|-|-|\n%s" % ("\n".join([
            f"|MAPE|{mape}|",
            f"|MAE|{mae}|",
            f"|MSE|{mse}|"
        ])))
    
  
def corr_matrix(data,writer):
    # Compute the correlation matrix
    correMtr = data.corr()
    mask = np.array(correMtr)
    mask[np.tril_indices_from(mask)] = False # the correlation matrix is symmetric
    # Heat map for correlation matrix
    fig,ax = plt.subplots(figsize=(16,16))
    sns.set_style("white")
    sns.heatmap(correMtr,mask=mask,vmin=-1.0,vmax=1.0,square=True,annot=True,fmt="0.2f",ax=ax)
    ax.set_title('Correlation matrix of attributes')
    plt.show()
    
    writer.add_figure("Correlation Matrix", fig)
    weight_corr = correMtr['PREORE_VAKI-Weight [g]'].drop('PREORE_VAKI-Weight [g]').sort_values()

    # Plot as a vertical bar chart
    fig,ax = plt.subplots(figsize=(8, 10))
    sns.barplot(y=weight_corr.index, x=weight_corr.values, hue=weight_corr.index, legend=False)
    plt.xlabel("Correlation with PREORE_VAKI-Weight [g]")
    plt.ylabel("Features")
    plt.title("Correlation of Features with PREORE_VAKI-Weight [g]")
    plt.show()
    writer.add_figure("Correlation Vector of PREORE_VAKI-Weight", fig)
    # Heat map for correlation matrix

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
    rf_pred = best_rf.predict(x_test)
    
    # Scale back predictions if required
    # if args.transformFlag:
    #     rf_pred = np.expm1(args.scaler_y.inverse_transform(rf_pred.reshape(-1, 1)))
    #     y_test = np.expm1(args.scaler_y.inverse_transform(y_test.reshape(-1, 1)))
    # else:
    #     rf_pred = args.scaler_y.inverse_transform(rf_pred.reshape(-1, 1))
    #     y_test = args.scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Compute and log evaluation metrics
    mse = mean_squared_error(y_test, rf_pred)
    mae = mean_absolute_error(y_test, rf_pred)
    mape = mean_absolute_percentage_error(y_test, rf_pred) * 100
    print(f"Random Forest MSE: {mse}")
    print(f"Random Forest MAE: {mae}")
    print(f"Random Forest MAPE: {mape}%")
    # Log results to TensorBoard
    writer.add_text("Random Forest Model Performance", 
                    f"|Metric|Value|\n|-|-|\n|MSE|{mse}|\n|MAE|{mae}|\n|MAPE|{mape}%|")
   
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in data.columns])
    writer.add_text("4: Feature Names", feature_table)
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
    plots = Custom_plots(rf_pred, y_test,writer)
    plots.plot_actual_vs_predicted_scatter()
    plots.plot_predictions()
    plots.plot_predictions_with_hist()
    plots.plot_residuals()
    writer.close()


def train(args, data):
    writer = SummaryWriter(args.path)
    writer.add_text(
        "3: Hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    x, y, data, fishweight_col_indx = prepare_data(args, data)
    samples = int(x.shape[0] / args.timesteps)
    x = np.array(x)
    y = np.array(y)
    x = x[:samples * args.timesteps].reshape(samples, args.timesteps, x.shape[1])
    y = y[:samples * args.timesteps].reshape(samples, args.timesteps)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    x_test_contained_fishWeight = x_test.copy()
    x_test = np.delete(x_test, fishweight_col_indx, axis=2)
    x_train = np.delete(x_train, fishweight_col_indx, axis=2)
    data = data.drop(columns=["Fish_Weight"])
    
    modelClass = ModelClass(args.timesteps, x.shape[2]-1, args.dropout, args.learning_rate)
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
    print(f"Test MAPE: {test_results[1]}")
    print(f"Test MAE: {test_results[2]}")
    print(f"Test MSE: {test_results[3]}")
    writer.add_text("1: Evaluation metrics using model.evaluate",
    "|Test Metric|Value|\n|-|-|\n%s" % ("\n".join([
        f"|MAPE|{test_results[1]}|",
        f"|MAE|{test_results[2]}|",
        f"|MSE|{test_results[3]}|"
    ])))
    
    # Predictions vs Actual values plot
    original_x_test = x_test.reshape(-1, x_test.shape[2])
    original_y_test = y_test.reshape(-1)
    
    x_reshaped = x_test_contained_fishWeight.reshape(-1, x_test_contained_fishWeight.shape[2])
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
    compute_metrics(np.array(predicted_values), np.array(actual_values),writer)
    
    plots = Custom_plots(np.array(predicted_values), np.array(actual_values),writer)
    plots.plot_actual_vs_predicted_scatter()
    plots.plot_predictions()
    plots.plot_predictions_with_hist()
    plots.plot_residuals()
    
    feature_table = "|Features|\n|-|\n"
    feature_table += "\n".join([f"|{name}|" for name in data.columns])
    writer.add_text("4: Feature Names", feature_table)

    plt.close(fig)
    writer.close()
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
        
    compute_metrics(Fish_Weight_Predictedby_Math_model, np.array(actual_values))
    plots = Custom_plots(np.array(Fish_Weight_Predictedby_Math_model), np.array(actual_values),writer)
    plots.plot_actual_vs_predicted_scatter()
    plots.plot_predictions()
    plots.plot_predictions_with_hist()
    plots.plot_residuals()
        
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
            
        run_name += f"({formatted_datetime})"

        if args.prediction_Method=="Random_Forest":
            run_name = "RF_" + run_name
            args.path = f"data/Runs/{run_name}"
            train_random_forest(args, data_all)
        else:
            run_name = str(args.timesteps)+"_"+args.prediction_Method
            args.path = f"data/Runs/{run_name}"
            args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
            train(args,data_all)

    
    # for i in range(0,len(data)-1):
    #     print("RNN on data of each time window "+str(i+1)+"___________________________________")
    #     data_all = data[i]['data_contextual_weight']
    #     data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp","Fish_Weight"],axis=1)
    #     rnn = RNN(data_all)
    #     rnn.train()
    


    
   
            
    
    
