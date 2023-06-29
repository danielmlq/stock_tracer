#!/bin/bash

config_file=$1
if [ -z $config_file ]; then
    echo "Need provide config file."
    exit 99
fi

source $config_file

echo "Step 1 -> Generating statistics raw data for all the input stocks for today."
python3 $STATISTICS_MODEL_PYTHON_DIR $EXISTING_STOCKS $STATISTICS_STOCK_OUTPUT
echo "Finished running statistics for all the input stocks"


echo "Step 2 -> Storing statistics to database"
IFS=',' read -r -a stocks_array <<< "$EXISTING_STOCKS"
for stock in "${stocks_array[@]}"
do
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -t -c "\copy ${stock}_information(day_of_week, information_date, open_price, high_price, low_price, close_price, change_value, change_percentage, up_or_down) from '$STATISTICS_STOCK_OUTPUT/$stock/raw.csv' WITH DELIMITER ','"
done
echo "Finished storing statistics to database"


echo "Step 3 -> Triggering predictions for all the input stocks"
rm -r $PREDICTIONS_STOCK_OUTPUT_DIR
mkdir -p $PREDICTIONS_STOCK_OUTPUT_DIR
python3 $PREDICTION_MODEL_PYTHON_DIR $config_file
echo "Finished predicting for all the input stocks"

echo "Step 4 -> Storing predictions to database"
for stock in "${stocks_array[@]}"
do
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -t -c "\copy ${stock}_predictions(open_price, open_rmse, open_plot_directory, high_price, high_rmse, high_plot_directory, low_price, low_rmse, low_plot_directory, close_price, close_rmse, close_plot_directory, prediction_date) from '$PREDICTIONS_STOCK_OUTPUT_DIR/$stock/predictions.csv' WITH DELIMITER ','"
done
echo "Finished storing predictions to database"