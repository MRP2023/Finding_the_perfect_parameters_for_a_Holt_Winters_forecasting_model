import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from collections import deque


# Other necessary imports...

# Database Connection
engine = create_engine('postgresql://stockeradminsimec:stock_admin_#146@localhost/stockerhubdb')

# Specify the schema and table name
schema_name = 'public'
table_name = 'company_price_companyprice'

# Fetch instrument codes from the database
instr_query = f"SELECT DISTINCT \"Instr_Code\" FROM {table_name}"
instr_codes = pd.read_sql(instr_query, engine)["Instr_Code"].tolist()

# Function for the program
def holt_winters(data, alpha, beta, gamma, periods):
    n = len(data)

    if n == 0:
        print("Error: Empty data passed to holt_winters function.")
        return []

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
        try:
            level = alpha * (data['closing_price'].iloc[i - n] - seasonality[i % n]) + (1 - alpha) * (level + trend)
            trend = beta * (level - level_old) + (1 - beta) * trend_old
            seasonality[i % n] = gamma * (data['closing_price'].iloc[i - n] - level) + (1 - gamma) * seasonality[i % n]
            forecast_linked_list.append(level + i * trend + seasonality[i % n])
        except Exception as e:
            print(f"Error at i={i}: {e}")
            break

    return forecast_linked_list


selected_columns = ['mkt_info_date', 'closing_price']

# Required Parameter Ranges
alpha_range = np.arange(0.1, 1.0, 0.1)
beta_range = np.arange(0.1, 1.0, 0.1)
gamma_range = np.arange(0.1, 1.0, 0.1)
periods_to_forecast = 237

best_params = None
best_rmse = float('inf')

for alpha in alpha_range:
    for beta in beta_range:
        for gamma in gamma_range:
            total_rmse = 0.0

            for target_instr_code in instr_codes:
                # Fetch data and prepare for forecasting...
                query = f"SELECT {', '.join(selected_columns)} FROM {table_name} WHERE \"Instr_Code\" = '{target_instr_code}' AND closing_price != 0"
                df1 = pd.read_sql(query, engine)
                closing_prices = df1['closing_price']
                df = df1.head(365)

                # Forecast using current parameters
                forecast_linked_list = holt_winters(df, alpha, beta, gamma, periods_to_forecast)

                # Extract actual values
                actual_values = closing_prices.values

                # Evaluate forecast using RMSE
                # Check if both actual_values and forecast_linked_list have at least one sample
                if len(actual_values) > 0 and len(forecast_linked_list) > 0:
                    # Evaluate forecast using RMSE
                    rmse = np.sqrt(mean_squared_error(actual_values[-periods_to_forecast:], forecast_linked_list))
                    total_rmse += rmse

            # Average RMSE across all instrument codes
            avg_rmse = total_rmse / len(instr_codes)

            # Update best parameters if current combination is better
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = (alpha, beta, gamma)

# Print the best parameters and best RMSE
print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)
