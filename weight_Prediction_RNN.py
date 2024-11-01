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



@dataclass
class Args:
    
    verbos = 0
    epochs: int = 200
    batch_size: int = 32  # Increased batch size
    validation_split: float = 0.2
    timesteps: int = 3 # Increased timesteps
    patience: int = 10  # Reduced patience for early stopping
    dropout: float = 0.2  # Increased dropout for regularization
    learning_rate: float = 0.001  # Slightly higher learning rate for faster convergence
    transformFlag: bool() = True
    displayCorrMatrix = True
    feature_names = []
    # Scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    root = 'data/Preore_Dataset/'
    path=""
    model_file = ""
    
    


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
    data['Entrance_timestamp'] = pd.to_datetime(data['Entrance_timestamp'])
    data['year'] = data['Entrance_timestamp'].dt.year
    data['month'] = data['Entrance_timestamp'].dt.month
    data['day_of_week'] = data['Entrance_timestamp'].dt.day_of_week
    data['day_of_month'] = data['Entrance_timestamp'].dt.days_in_month
    data['hour'] = data['Entrance_timestamp'].dt.hour
    data['minute'] = data['Entrance_timestamp'].dt.minute
    data['second'] = data['Entrance_timestamp'].dt.second
    data = data.drop(["Entrance_timestamp"], axis=1)
    data = data.drop(data.columns[data.columns.str.contains('EXIT')], axis=1)
    
    # data = data.drop(["Energy_Acquisition(A)"], axis=1)
    # data = data.drop(["Catabolic_component(C)"], axis=1)
    # data = data.drop(["Somatic_tissue_energy_content(Epsilon)"], axis=1)
    data = data
    x = data.drop(columns=["PREORE_VAKI-Weight [g]"])
    # Optional: Apply log transformation to target
    y = data["PREORE_VAKI-Weight [g]"].values

    if args.transformFlag:
        y = np.log1p(data["PREORE_VAKI-Weight [g]"].values)  # Use log1p for stability        
    # Apply scaling
    x = args.scaler_x.fit_transform(x)
    y = args.scaler_y.fit_transform(y.reshape(-1, 1))
    
    return x, y

def compute_metrics(predicted_values,actual_values):
    

    
    # Calculate loss, MSE, MAE, and MAPE
    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    
def create_model( x_train_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2]))))
    model.add(Dropout(args.dropout))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(args.dropout))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(args.dropout))
    model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error', metrics=['mape', 'mae', 'mse'])
    return model

   
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
    
    writer.add_figure("Correlation Matrix", fig, global_step=0)
    weight_corr = correMtr['PREORE_VAKI-Weight [g]'].drop('PREORE_VAKI-Weight [g]').sort_values()

    # Plot as a vertical bar chart
    fig,ax = plt.subplots(figsize=(8, 10))
    sns.barplot(y=weight_corr.index, x=weight_corr.values, hue=weight_corr.index, legend=False)
    plt.xlabel("Correlation with PREORE_VAKI-Weight [g]")
    plt.ylabel("Features")
    plt.title("Correlation of Features with PREORE_VAKI-Weight [g]")
    plt.show()
    writer.add_figure("Correlation Vector of PREORE_VAKI-Weight", fig, global_step=0)
    # Heat map for correlation matrix

    
def train(args, data):
    writer = SummaryWriter(args.path)
    
    # Log hyperparameters
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    x, y = prepare_data(args, data)
    samples = int(x.shape[0] / args.timesteps)
    x = np.array(x)
    y = np.array(y)
    
    x = x[:samples * args.timesteps].reshape(samples, args.timesteps, x.shape[1])
    y = y[:samples * args.timesteps].reshape(samples, args.timesteps)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    
    model = create_model(x_train.shape)
    
    checkpointer = ModelCheckpoint(filepath=args.model_file, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
        validation_split=args.validation_split, callbacks=[checkpointer, early_stopping], verbose=args.verbos
    )
    
    # Log training metrics
    log_metrics(writer, history)
    
    # Plot loss and metrics and log to TensorBoard
    for metric in ['loss', 'mse', 'mae', 'mape']:
        fig, ax = plt.subplots()
        ax.plot(history.history[metric], label=f'Training {metric.upper()}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.upper())
        ax.legend()
        
       
        writer.add_figure(f"Training/{metric.upper()}", fig, global_step=0)

        plt.close(fig)  # Close the plot to save memory
    
    # Final evaluation and predictions
    test_results = model.evaluate(x_test, y_test)
    print(f"Test MAPE: {test_results[1]}")
    print(f"Test MAE: {test_results[2]}")
    print(f"Test MSE: {test_results[3]}")
    
    # Predictions vs Actual values plot
    original_x_test = x_test.reshape(-1, x_test.shape[2])
    original_y_test = y_test.reshape(-1)
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
    
    # Plot and log predictions vs actual values
    fig, ax = plt.subplots()
    ax.plot(actual_values, label='Actual Values')
    ax.plot(predicted_values, label='Predicted Values', linestyle='--')
    ax.set_xlabel("Samples")
    ax.set_ylabel("Weight [g]")  # Adjust to your target variable's unit
    ax.legend()
    

    writer.add_figure("Prediction vs Actual", fig, global_step=0)

    plt.close(fig)
    
    writer.close()
    
    if args.displayCorrMatrix:
        corr_matrix(data,writer)
        
if __name__ == "__main__":
    
    args = tyro.cli(Args)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{formatted_datetime}"
    args.path = f"data/Runs/{run_name}"
    args.model_file = args.path + '/fish_weight_prediction_model.hdf5'
    
    with open(args.root + 'results/dynamic_individual_weight.pkl', 'rb') as file:
        data = pickle.load(file)
    data_all = data['data_contextual_weight']
    data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp","Fish_Weight"],axis=1)
    print("RNN on All data_____________________________________________________") 
    
    data_all = data['data_contextual_weight']
    data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp","Fish_Weight"],axis=1)
    train(args,data_all)
    
    # for i in range(0,len(data)-1):
    #     print("RNN on data of each time window "+str(i+1)+"___________________________________")
    #     data_all = data[i]['data_contextual_weight']
    #     data_all = data_all.drop(["index","Unnamed: 0","Exit_timestamp","observed_timestamp","Fish_Weight"],axis=1)
    #     rnn = RNN(data_all)
    #     rnn.train()
    


    
   
            
    
    
