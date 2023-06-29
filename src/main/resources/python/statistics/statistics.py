import os
import sys
import shutil
from datetime import date, timedelta
import pandas_datareader as web
import numpy as np

class statistics:

    def truncate_raw_float_columns(input_df):
        input_df['Open'] = input_df.apply(lambda row: np.round(row.Open, 2), axis=1)
        input_df['High'] = input_df.apply(lambda row: np.round(row.High, 2), axis=1)
        input_df['Low'] = input_df.apply(lambda row: np.round(row.Low, 2), axis=1)
        input_df['Close'] = input_df.apply(lambda row: np.round(row.Close, 2), axis=1)

    def add_date_column(input_df):
        input_df['Date'] = input_df.index

    def calculate_change(input_df):
        input_df['Change'] = input_df.apply(lambda row: row.Close - row.Open, axis=1)

    def calculate_change_percentage(input_df):
        input_df['Change_Percentage'] = input_df.apply(lambda row: np.round(row.Change/row.Open * 100, 2), axis=1)

    def go_up_or_down(input_df):
        input_df['Up'] = np.where(input_df['Change'] >= 0, 'Positive', 'Negative')

    # Monday=0, Sunday=6
    def calculate_day_of_week(input_df):
        input_df['Day_Of_Week'] = input_df.index.dayofweek

    def get_week_day(input_df):
        switcher = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        input_df['Week_Day'] = input_df.apply(lambda row: switcher.get(row.Day_Of_Week), axis=1)

    def generate_raw_output(input_df, output_csv_file):
        # Delete legacy data if exists and create new empty output file for storing results.
        if os.path.exists(output_csv_file):
            os.remove(output_csv_file)

        open(output_csv_file, 'a')
        input_df.tail(1).to_csv(output_csv_file, index=False, header=False)

    if __name__ == '__main__':
        stock_to_run = sys.argv[1]
        output_dir = sys.argv[2]

        start_date = (date.today() - timedelta(days=2)).__format__('%Y-%m-%d')
        end_date = date.today().__format__('%Y-%m-%d')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        stock_array = stock_to_run.split(',')
        for stock in stock_array:
            sub_output_dir = "{}/{}/".format(output_dir, stock)
            if os.path.exists(sub_output_dir):
                shutil.rmtree(sub_output_dir)
            os.mkdir(sub_output_dir)

            input_df = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)

            truncate_raw_float_columns(input_df)
            add_date_column(input_df)
            # Calculate raw data
            calculate_change(input_df)
            calculate_change_percentage(input_df)
            go_up_or_down(input_df)
            calculate_day_of_week(input_df)
            get_week_day(input_df)

            base_df = input_df.filter(['Week_Day', 'Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Change_Percentage', 'Up'])
            generate_raw_output(base_df, "{}/raw.csv".format(sub_output_dir))