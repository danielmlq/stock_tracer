import os
import sys
from configobj import ConfigObj
import shutil
import math
import numpy as np
from datetime import date, timedelta
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class Module:
    def truncate_raw_float_columns(input_df):
        input_df['Open'] = input_df.apply(lambda row: np.round(row.Open, 2), axis=1)
        input_df['High'] = input_df.apply(lambda row: np.round(row.High, 2), axis=1)
        input_df['Low'] = input_df.apply(lambda row: np.round(row.Low, 2), axis=1)
        input_df['Close'] = input_df.apply(lambda row: np.round(row.Close, 2), axis=1)

    def calculate_prediction(input_df, column_name, output_file, fig_output_dir, current_date):
        price = input_df.iloc[-1][column_name]
        rmse = 9999

        if price <= 300:
            rmse_limit = 3
        elif price > 300 and price <= 500:
            rmse_limit = 4
        elif price > 500 and price <= 1000:
            rmse_limit = 5
        elif price > 1000 and price <= 2000:
            rmse_limit = 6
        else:
            rmse_limit = 7

        while rmse > rmse_limit:
            # Create a new dataframe with only 'Close' column
            data = input_df.filter([column_name])

            # Convert the dataframe to a numpy array
            dataset = data.values

            # Get the number of rows to train the model on
            training_data_len = math.ceil(len(dataset) * .95)

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Create the training dataset, the scaled training dataset
            train_data = scaled_data[0:training_data_len, :]
            # Split the training data into x_train and y_train datasets
            x_train = []
            y_train = []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
                if i <= 60:
                    print(x_train)
                    print(y_train)
                    print()

            # Convert x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshap the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_train.shape

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(x_train, y_train, batch_size=1, epochs=1)

            # Create the testing dataset
            # Create a new array containing scaled values from index 1705 to 2130
            test_data = scaled_data[training_data_len - 60: , :]
            # Create the test dataset x_test and y_test
            x_test = []
            y_test = dataset[training_data_len: , :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])

            # Convert the data to a np array
            x_test = np.array(x_test)

            # Reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Get the model predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Get the root mean squared error (RMSE)
            rmse = np.sqrt(np.mean(predictions - y_test)**2)

            # In the future, leverage valid dataframe to compare the actual value and prediction value
            # to calculate the gap to further reduce the error.
            train = data[: training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions

            # Get the last 60 days of close price and convert to array from dataframe
            last_60_days = data[-60:].values

            # Scale the values in the array from 0 to 1
            last_60_days_scaled = scaler.transform(last_60_days)

            # Create an empty list
            prediction_test = []
            prediction_test.append(last_60_days_scaled)
            X_test = np.array(prediction_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Get prediction for scaled price
            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)

        # Visualize data and save
        plt.figure(column_name, figsize=(16, 8))
        plt.title("Prediction vs Actual")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("{} Price USD ($)".format(column_name), fontsize=18)
        plt.plot(train[column_name])
        plt.plot(valid[[column_name, 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

        if not os.path.exists(fig_output_dir):
            os.mkdir(fig_output_dir)

        fig_dir = "{}/{}".format(fig_output_dir, current_date)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        fig_path = "{}/{}_plot.png".format(fig_dir, column_name)
        plt.savefig(fig_path)

        output_pred = np.round(np.float(pred_price[0][0]), 2)
        output_rmse = np.round(rmse, 2)

        f = open(output_file, 'a')
        line = "{},{},{},".format(output_pred, output_rmse, fig_path)
        f.write(line)
        f.flush()

    if __name__ == '__main__':

        config_file = sys.argv[1]
        cfg = ConfigObj(config_file)
        start_date = (date.today() - timedelta(days=365*8)).__format__('%Y-%m-%d')
        end_date = date.today().__format__('%Y-%m-%d')
        output_dir = cfg.get('PREDICTIONS_STOCK_OUTPUT_DIR')
        stock_to_run = cfg.get('EXISTING_STOCKS')
        fig_output_dir = cfg.get('PREDICTIONS_FIG_OUTPUT_DIR')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        for stock in stock_to_run:
            sub_output_dir = "{}/{}".format(output_dir, stock)
            output_file = "{}/predictions.csv".format(sub_output_dir)
            stock_figure_output_dir = "{}/{}".format(fig_output_dir, stock)

            if os.path.exists(sub_output_dir):
                shutil.rmtree(sub_output_dir)
            os.mkdir(sub_output_dir)

            df = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)
            truncate_raw_float_columns(df)

            columns = ['Open', 'High', 'Low', 'Close']
            for col in columns:
                calculate_prediction(df, col, output_file, stock_figure_output_dir, end_date)

            f = open(output_file, 'a')
            f.write(end_date)
            f.flush()