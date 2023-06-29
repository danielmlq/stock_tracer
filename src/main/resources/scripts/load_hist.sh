#!/bin/bash

config_file=$1
if [ -z $config_file ]; then
    echo "Need provide config file."
    exit 99
fi

source $config_file

# Start of definition of functions
function setup_new_stock_environment(){
    stock_name=$1

    prediction_env_sequence_creation_query="create sequence ${stock_name}_prediction_id_seq"

    prediction_env_creation_query="create table if not exists ${stock_name}_predictions (
                                        prediction_id Bigint DEFAULT nextval('${stock_name}_prediction_id_seq'),
                                        prediction_date Date NOT NULL,
                                        open_price Float NOT NULL,
                                        open_rmse Float NOT NULL,
                                        open_plot_directory varchar(1000) NOT NULL,
                                        high_price Float NOT NULL,
                                        high_rmse Float NOT NULL,
                                        high_plot_directory varchar(1000) NOT NULL,
                                        low_price Float NOT NULL,
                                        low_rmse Float NOT NULL,
                                        low_plot_directory varchar(1000) NOT NULL,
                                        close_price Float NOT NULL,
                                        close_rmse Float NOT NULL,
                                        close_plot_directory varchar(1000) NOT NULL,
                                        PRIMARY KEY (prediction_id)
                                    )"

    statistics_env_sequence_creation_query="create sequence ${stock_name}_information_id_seq"

    statistics_env_creation_query="create table if not exists ${stock_name}_information (
                                        information_id Bigint DEFAULT nextval('${stock_name}_information_id_seq'),
                                        day_of_week varchar(20) NOT NULL,
                                        information_date Date NOT NULL,
                                        open_price Float NOT NULL,
                                        high_price Float NOT NULL,
                                        low_price Float NOT NULL,
                                        close_price Float NOT NULL,
                                        change_value Float NOT NULL,
                                        change_percentage Float NOT NULL,
                                        up_or_down varchar(20),
                                        PRIMARY KEY (information_id)
                                    )"
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -c "$prediction_env_sequence_creation_query"
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -c "$prediction_env_creation_query"
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -c "$statistics_env_sequence_creation_query"
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -c "$statistics_env_creation_query"
}

# End of definition of functions

echo "Step 1 -> Setting up database environment for all the input stocks if not exist"
IFS=',' read -r -a stocks_array <<< "$NEW_STOCKS"
for stock in "${stocks_array[@]}"
do
    setup_new_stock_environment $stock
done
echo "Finished setting up database environment for all the input stocks if not exist"


echo "Step 2 -> Trigger statistics for all the input stocks"
python3 $LOAD_HISTORICAL_PYTHON_DIR $NEW_STOCKS $LOAD_HIST_RAW_OUTPUT_DIR
echo "Finished running statistics for all the input stocks"


echo "Step 3 -> Storing statistics to database"
for stock in "${stocks_array[@]}"
do
    psql "host=$DB_HOST dbname=$DB_NAME user=$DB_USER password=$DB_PWD" -t -c "\copy ${stock}_information(day_of_week, information_date, open_price, high_price, low_price, close_price, change_value, change_percentage, up_or_down) from '$LOAD_HIST_RAW_OUTPUT_DIR/$stock/raw.csv' WITH DELIMITER ','"
done
echo "Finished storing statistics to database"