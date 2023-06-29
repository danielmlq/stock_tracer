import sys
import math
import numpy as np
from datetime import date
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class Module:

    def calculate_prediction(input_df, column_name):
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

        # Calculate difference and percentage
        valid['Difference'] = valid.apply(lambda row: row.Close - row.Predictions, axis=1)
        valid['Difference_Percentage'] = valid.apply(lambda row: "{}%".format((row.Close - row.Predictions) / row.Close * 100), axis=1)

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.savefig('/Users/danielmeng/Downloads/Legend/test.png')

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
        valid.to_csv('/Users/danielmeng/Downloads/Legend/test.csv', index=False, header=True)
        f = open('/Users/danielmeng/Downloads/Legend/test.csv', 'a')
        f.write("\nrmse = {}, prediction = {}\n".format(rmse, pred_price[0][0]))


    if __name__ == '__main__':
        stock_to_run = sys.argv[1]
        start_date = sys.argv[2]
        end_date = date.today().__format__('%Y-%m-%d')

        stock_array = stock_to_run.split(',')

        for stock in stock_array:
            df = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)
            columns = ['Close']
            for col in columns:
                calculate_prediction(df, col)