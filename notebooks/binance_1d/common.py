import os
from math import sqrt

import keras.backend as K
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import optuna
import pandas as pd
from keras.callbacks import Callback
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from statsmodels.tsa.vector_ar.var_model import VAR

import sys
sys.path.append("../../airflow/dags/common/models")

from models import build_model, get_splits


class OptunaPruneCallback(Callback):
    def __init__(self, trial):
        super(Callback, self).__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs={}):
        if self.trial:
            self.trial.report(logs["val_mean_absolute_percentage_error_keras"], epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def get_dataframe(add_sma_columns: bool = False):
        
    folder = os.path.join("../../airflow/assets/binance_1d")
    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(folder, file), skiprows=1, parse_dates=['Date']))

    # Step 1: Convert "date" column to datetime in all dataframes
    for df in dfs:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors="coerce")

    # Step 2: Find the oldest and newest dates across all dataframes
    all_dates = [df['Date'] for df in dfs]
    all_dates_flat = [date for sublist in all_dates for date in sublist if not pd.isnull(date)]

    oldest_date = '2019-01-01'
    newest_date = max(all_dates_flat)

    # Step 3: Create a new dataframe with the date range
    date_range = pd.date_range(start=oldest_date, end=newest_date, freq='D')  # Daily frequency
    merged_df = pd.DataFrame({'Date': date_range})

    # Step 4: Add "close" and "Volume USDT" columns from each dataframe to the merged_df using list comprehension
    for df in dfs:
        try:
            ticker = df['Symbol'].iloc[0]  # Assuming each dataframe has a "symbol" column
            close_col_name = f'close_{ticker}'
            volume_col_name = f'Volume USDT_{ticker}'  # Replace with the actual column name if it's different in your data

            df = df.set_index('Date').sort_index()

            # Create DataFrames with the "date" and "close" columns
            close_data = df[df.index.isin(date_range)][['Close']]
            close_data.rename(columns={'Close': close_col_name}, inplace=True)

            # Merge the "close_data" into the "merged_df"
            merged_df = pd.merge(merged_df, close_data, left_on='Date', right_index=True, how='left')

        except ValueError as e:
            print(f'Error on coin {ticker}: {e}')

    if add_sma_columns:
        merged_df = pd.concat([merged_df['Date'], calculate_sma(merged_df.iloc[:, 1:])], axis=1)

    return merged_df


def get_clustered_dataframes(add_sma_columns: bool = False,
                             experiment_id: str = "3",
                             run_id: str = "20261760bd664b6c82a92c273a51205c"):

    merged_df = get_dataframe()
    experiment = _get_experiments_from_mlflow(experiment_id, run_id)

    # Use eval() to convert the string to a list of tuples
    data_list = eval(experiment["params.Cluster_Labels"].iloc[0])

    cripto_clusters = {}

    for key, value in data_list:
        if value not in cripto_clusters:
            cripto_clusters[value] = []
        cripto_clusters[value].append(key)

    assert len(cripto_clusters[0] + cripto_clusters[1] + cripto_clusters[2]) == len(data_list)

    clusters_data = {}

    # loop on key and value of cripto_clusters
    for cluster, criptos in cripto_clusters.items():
        _criptos = criptos + ['Date']
        clusters_data[cluster] = merged_df[_criptos]
        if add_sma_columns:
            clusters_data[cluster] = pd.concat([merged_df['Date'], calculate_sma(clusters_data[cluster][criptos])], axis=1)

    return clusters_data


def calculate_sma(
            cripto_data: DataFrame,
            sma_window_sizes: list = [5,10,20,50,100,200],
            min_periods: int = 1
            ):
    
    """
    This function calculates the Simple Moving Average (SMA) for each window size
    and returns the indexed data.
    
    Parameters
    ----------
    cripto_data : DataFrame
        Cripto data to be indexed.
    sma_window_sizes : list
        List of window sizes to calculate the SMA.
    min_periods : int
        Minimum number of observations in window required to have a value.

    """

    df = DataFrame(cripto_data)

    for cripto in df.columns:
        cripto_name = cripto.split('_')[1]
        # Calculate Simple Moving Average (SMA) for each window size
        for window_size in sma_window_sizes:
                sma_column = f'sma_{window_size}_{cripto_name}'
                df[sma_column] = df[cripto].rolling(window=window_size, min_periods=1).mean()

    return df


def _get_experiments_from_mlflow(experiment_id: str, run_id: str):
    experiments = mlflow.search_runs(experiment_ids=experiment_id)
    return experiments.loc[experiments['run_id'] == run_id]


def evaluate_best_coin(
    coin,
    data,
    best_params,
    output_shape = None
):
    """
    Evaluate the best model for a given coin
    
    :param coin: The coin to be used for training
    :param data: The data to be used for training
    :param best_params: The best hyperparameters found by Optuna
    """

    # Train the final model with the best hyperparameters
    # Define the search space for hyperparameters
    num_layers = best_params['num_layers']
    units_per_layer = [best_params[f'units_layer_{i}'] for i in range(num_layers)]
    sequence_length = best_params['sequence_length']
    learning_rate = best_params['learning_rate']
    dropout_rate = best_params['dropout_rate']
    min_max_scaling = best_params['min_max_scaling']
    layer_type = best_params['layer_type']
    optimizer = best_params['optimizer']
    activation = best_params['activation']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']

    layer_class = keras.layers.LSTM if layer_type == 'LSTM' else keras.layers.SimpleRNN 

    if min_max_scaling == 1:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.array(data))

    X_train, X_test, X_val, y_train, y_test, y_val = get_splits(data, sequence_length, output_shape)
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    with mlflow.start_run(
        run_name=f"Training_best_model_{coin}"
    ) as run:
    
        mlflow.log_params({
            'coin': coin,
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'sequence_length': sequence_length,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'min_max_scaling': min_max_scaling,
            'layer_type': layer_type,
            'optimizer': optimizer,
            'activation': activation,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'output_shape': output_shape
        })

        model = build_model(
            data,
            units_per_layer,
            sequence_length,
            learning_rate,
            dropout_rate,
            layer_class,
            optimizer,
            activation,
            weight_decay,
            output_shape
        )

        model.fit(X_train, y_train, epochs=100, batch_size=batch_size)

        preds = model.predict([X_test])

        # Min max scaling first column
        if min_max_scaling == 1:
            # The min-max scaler needs the same number of columns that it was fitted on
            # So we create a new array with the same shape as the predictions array
            # and then we replace the first column with the predictions
            _preds = np.zeros(shape=(len(y_test), data.shape[1]))
            _preds[:,0] = preds[:, 0]
            preds = scaler.inverse_transform(_preds)[:, 0]

            _y_test = np.zeros(shape=(len(y_test), data.shape[1]))
            _y_test[:,0] = y_test[:, 0]
            y_test = scaler.inverse_transform(_y_test)[:, 0]
            
        register_training_experiment(y_test, preds, model_name = layer_class, coin = coin)


def register_training_experiment(
    data,
    predictions,
    model_name = None,
    coin = None,
    x_axis = None,
    params = {},
):

    if x_axis is None:
        x_axis = range(len(data))
        
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, data, label='original')
    plt.plot(x_axis, predictions, label='predicted')
    plt.legend()
    plt.title(f"{coin}")
    plt.show()

    with mlflow.start_run(run_name='Evaluation', nested=True):
        mlflow.log_params({
            'model': model_name,
            'coin': coin,
            **params
        })

        for days in [len(data), 90, 30, 7]:
            mse = mean_squared_error(data[-days:], predictions[-days:])
            rmse = sqrt(mse)
            mape = mean_absolute_percentage_error(data[-days:], predictions[-days:])
            print(f"Metrics for {days} days: {coin} MSE: {mse}, RMSE: {rmse}", f"MAPE: {mape}")

            suffix = 'all' if days == len(data) else days
            mlflow.log_metrics({
                f'mse_{suffix}': mse,
                f'rmse_{suffix}': rmse,
                f'mape_{suffix}': mape
            })


        for days in [len(data), 90, 30, 7]:
            mse = mean_squared_error(data[:days], predictions[:days])
            rmse = sqrt(mse)
            mape = mean_absolute_percentage_error(data[:days], predictions[:days])
            print(f"Metrics for {days} days: {coin} MSE: {mse}, RMSE: {rmse}", f"MAPE: {mape}")

            suffix = 'all' if days == len(data) else days
            mlflow.log_metrics({
                f'mse_{suffix}_v2': mse,
                f'rmse_{suffix}_v2': rmse,
                f'mape_{suffix}_v2': mape
            })


def objective(trial, data, coins, output_shape = None):
    """
    Define an objective function to be minimized by using Optuna.
    The hyperparameters are:
    - Number of layers
    - Number of units per layer
    - Sequence length
    - Learning rate
    - Dropout rate
    - Min-max scaling
    - Layer type (LSTM or RNN)

    Implements the OptunaPruneCallback
    
    :param trial: An Optuna trial object
    :param data: The data to be used for training
    :param coin: The coin to be used for training

    :return: The mean absolute percentage error on the validation set
    """

    with mlflow.start_run() as run:
        mlflow.keras.autolog(log_models=False)

        # Define the search space for hyperparameters
        num_layers = trial.suggest_int('num_layers', 1, 4)
        units_per_layer = [trial.suggest_int(f'units_layer_{i}', 8, 32, 8) for i in range(num_layers)]
        sequence_length = trial.suggest_int('sequence_length', 1, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        min_max_scaling = trial.suggest_int('min_max_scaling', 0, 1)
        layer_type = trial.suggest_categorical('layer_type', ['LSTM', 'RNN'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.5)
        batch_size = trial.suggest_int('batch_size', 8, 32, 8)

        layer_class = keras.layers.LSTM if layer_type == 'LSTM' else keras.layers.SimpleRNN 

        mlflow.log_params({
            'coin': coins,
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'sequence_length': sequence_length,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'min_max_scaling': min_max_scaling,
            'layer_type': layer_type,
            'optimizer': optimizer,
            'activation': activation,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'output_shape': output_shape
        })

        if min_max_scaling == 1:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(np.array(data))

        model = build_model(
            data,
            units_per_layer,
            sequence_length,
            learning_rate,
            dropout_rate,
            layer_class,
            optimizer,
            activation,
            weight_decay,
            output_shape
        )

        # Split the data into training and validation sets
        X_train, _, X_val, y_train, _, y_val = get_splits(data, sequence_length, output_shape)

        # Train the model with early stopping
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val, y_val), 
            verbose=0,
            callbacks=[OptunaPruneCallback(trial=trial)]
        )

        # Evaluate the model on the validation set
        loss = model.evaluate(X_val, y_val)

        return loss[1]
