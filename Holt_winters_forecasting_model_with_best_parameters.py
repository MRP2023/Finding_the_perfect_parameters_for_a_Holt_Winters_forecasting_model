import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import matplotlib.dates as mdates
import time
from collections import deque

# Database Connection
engine = create_engine('postgresql://stockeradminsimec:stock_admin_#146@localhost/stockerhubdb')

# Specify the schema and table name
schema_name = 'public'
table_name = 'company_price_companyprice'

instr_codes = pd.read_sql(f"SELECT \"Instr_Code\" FROM public.company_price_companyprice group by \"Instr_Code\"", engine)["Instr_Code"].tolist()

engine_sc = engine
selected_columns = ['mkt_info_date', 'closing_price']

# Function for the program
def holt_winters(data, alpha, beta, gamma, periods):
    n = len(data)
    level = data['closing_price'].iloc[0]  # Use column name 'closing_price'
    trend = np.mean(data['closing_price'].iloc[1:n] - data['closing_price'].iloc[0:n - 1])
    seasonality = np.array([data['closing_price'].iloc[i] - level - trend * i for i in range(n)])
    forecast_linked_list = deque()

    for i in range(n, n + periods):
        # Adjust the loop range to avoid index out-of-bounds
        if i - n >= len(data):
            break  # Exit the loop if we reach the end of the data
        print(f"i: {i}, n: {n}, len(data): {len(data)}")

        # Calculate the forecast components
        level_old, trend_old = level, trend
        level = alpha * (data['closing_price'].iloc[i - n] - seasonality[i % n]) + (1 - alpha) * (level + trend)
        trend = beta * (level - level_old) + (1 - beta) * trend_old
        seasonality[i % n] = gamma * (data['closing_price'].iloc[i - n] - level) + (1 - gamma) * seasonality[i % n]
        forecast_linked_list.append(level + i * trend + seasonality[i % n])

    return forecast_linked_list


# Function for moving average
def moving_average(data, window_size):
    return data['closing_price'].rolling(window=window_size, min_periods=1).mean()


# Function for exponential moving average
def exponential_moving_average(data, alpha):
    return data['closing_price'].ewm(alpha=alpha, adjust=False).mean()


# Required Parameter
alpha = 0.1
beta = 0.1
gamma = 0.1
periods_to_forecast = 237

for target_instr_code in instr_codes:
    # Construct the SQL query to select data from the specified table and filter by name
    query = f"SELECT {', '.join(selected_columns)} FROM {table_name} WHERE \"Instr_Code\" = '{target_instr_code}' AND closing_price != 0"
    # Execute the query and load the results into a DataFrame
    df1 = pd.read_sql(query, engine_sc)
    closing_prices = df1['closing_price']
    df = df1.head(365)

    forecast_linked_list = holt_winters(df, alpha, beta, gamma, periods_to_forecast)
    print(instr_codes)
    print(f"Forecasted Closing Prices {target_instr_code}:")

    # Print the linked list values
    for value in forecast_linked_list:
        print(value)

    # Plot actual closing prices
    # plt.plot(df['mkt_info_date'], df['closing_price'], label='Actual Closing Prices', marker='o')

    # Plot forecasted closing prices
    forecast_dates = pd.date_range(start=df['mkt_info_date'].iloc[-1] + pd.DateOffset(days=1), periods=periods_to_forecast, freq='B')
    forecast_values = list(forecast_linked_list)

    print("Length of forecast_dates:", len(forecast_dates))
    print("Length of forecast_values:", len(forecast_values))

    print("First few elements of forecast_dates:", forecast_dates[:5])
    print("First few elements of forecast_values:", forecast_values[:5])

    # plt.plot(forecast_dates, forecast_values, linestyle='--', marker='o', color='red', label='Forecasted Closing Prices')
    #
    # # Customize the plot
    # plt.title(f'Holt-Winters Forecasting for "{target_instr_code}"')
    # plt.xlabel('Date')
    # plt.ylabel('Closing Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # time.sleep(3)

# New way to plot
# Sorting the historical and forecast dates and values
    df1['mkt_info_date'] = pd.to_datetime(df1['mkt_info_date'])
    df1 = df1.sort_values('mkt_info_date')
    actual_dates = df1['mkt_info_date']
    actual_values = df1['closing_price']

    df['smoothed_ma'] = moving_average(df, window_size=5)  # Adjust window_size as needed
    df['smoothed_ema'] = exponential_moving_average(df, alpha=0.2)  # Adjust alpha as needed

    # Plotting the results
    plt.figure(figsize=(10, 6))

    # Plot actual closing prices
    plt.plot(actual_dates, actual_values, label='Actual Closing Prices', marker='o')

    # Plot smoothed moving average
    # plt.plot(df['mkt_info_date'], df['smoothed_ma'], label='Moving Average', linestyle='--', marker='o', color='green')

    # Plot smoothed exponential moving average
    # plt.plot(df['mkt_info_date'], df['smoothed_ema'], label='Exponential Moving Average', linestyle='--', marker='o',
    #          color='orange')

    # Plot forecasted closing prices
    forecast_dates, forecast_values = zip(*sorted(zip(forecast_dates, forecast_values)))
    plt.plot(forecast_dates, forecast_values, linestyle='--', marker='o', color='red', label='Forecasted Closing Prices')

    # Customize the plot
    plt.title(f'Holt-Winters Forecasting for "{target_instr_code}"')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"NOV302023/Holt_winters_Adaptibe_model_using _Deque_with_spliting_actual_data/Holt-Winters Forecasting Difference with Actual Closing for {target_instr_code}.png")
    # plt.savefig(f"Holt_winters_Adaptibe_model_using _Deque_after_removing_noise_movingaverage_exponentialmovingaverage/29NOV2023/using_exponential_moving_average/Holt-Winters Forecasting for {target_instr_code}.png")
    # plt.savefig(f"Holt_winters_Adaptibe_model_using _Deque_after_removing_noise_movingaverage_exponentialmovingaverage/29NOV2023_Remove_Closing_Price_0/using_Moving_Average/Holt-Winters Forecasting for {target_instr_code}.png")
    plt.show()
    time.sleep(3)