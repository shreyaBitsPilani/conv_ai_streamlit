import os
import yfinance as yf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the folder where CSV files will be saved
save_folder = os.path.join(BASE_DIR, "..", "data", "raw")
# Create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Define the ticker symbol for Apple
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)

# Download financial statements: income statement, balance sheet, and cash flow
income_statement = ticker.financials
balance_sheet = ticker.balance_sheet
cash_flow = ticker.cashflow

# Function to filter columns by year from a DataFrame with timestamp columns
def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    # Convert the columns to datetime if not already
    new_cols = pd.to_datetime(df.columns)
    # Filter columns where the year matches the given year
    filtered_cols = [col for col, dt in zip(df.columns, new_cols) if dt.year == year]
    return df[filtered_cols]

# Filter the statements for 2023 and 2024
income_2023 = filter_by_year(income_statement, 2023)
income_2024 = filter_by_year(income_statement, 2024)

balance_2023 = filter_by_year(balance_sheet, 2023)
balance_2024 = filter_by_year(balance_sheet, 2024)

cashflow_2023 = filter_by_year(cash_flow, 2023)
cashflow_2024 = filter_by_year(cash_flow, 2024)

# Save DataFrames to CSV files in the specified folder
income_2023.to_csv(os.path.join(save_folder, "income_statement_2023.csv"))
income_2024.to_csv(os.path.join(save_folder, "income_statement_2024.csv"))
balance_2023.to_csv(os.path.join(save_folder, "balance_sheet_2023.csv"))
balance_2024.to_csv(os.path.join(save_folder, "balance_sheet_2024.csv"))
cashflow_2023.to_csv(os.path.join(save_folder, "cash_flow_2023.csv"))
cashflow_2024.to_csv(os.path.join(save_folder, "cash_flow_2024.csv"))

print("Financial statements saved in:", save_folder)
