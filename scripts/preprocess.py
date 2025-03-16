# preprocess.py
"""
Reads raw CSV financial data from Yahoo Finance (e.g. income statements,
balance sheets, cash flow statements for 2023/2024), cleans/structures them,
and outputs a well-structured JSON with fields like:
   { "year": <int>, "finance_parameter": <str>, "value": <float> }
"""

import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_clean_data(raw_data_dir):
    """
    Loads CSV files from raw_data_dir, cleans and structures them,
    returns a list of records (dict).
    """
    all_records = []
    
    # Example: Suppose your 6 CSVs have columns: [Parameter, Value] or similar
    # Adjust column names to match your actual CSV structure
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(raw_data_dir, filename)
            
            # Derive the year from filename (e.g. "income_statement_2023.csv")
            # This is a naive approach; adapt to your naming convention
            # e.g. split on underscore and use last token before ".csv"
            # that might be "2023" or "2024", etc.
            parts = filename.replace(".csv","").split("_")
            year = None
            for part in parts:
                if part.isdigit():
                    year = int(part)  # pick up 2023 or 2024
            
            df = pd.read_csv(filepath)
            # Example: rename columns to a standard set
            df.columns = ["finance_parameter", "value"]
            
            # Clean up data—drop NAs, convert numeric types, etc.
            df.dropna(inplace=True)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.dropna(inplace=True)
            
            # Build list of {year, finance_parameter, value} dicts
            for _, row in df.iterrows():
                record = {
                    "year": year,
                    "finance_parameter": row["finance_parameter"],
                    "value": row["value"]
                }
                all_records.append(record)
    
    return all_records

def save_to_json(records, out_path):
    """
    Saves the list of records (dict) to a single JSON file.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

if __name__ == "__main__":
    raw_data_dir = os.path.join(BASE_DIR, "..", "data", "raw")
    raw_data_dir = os.path.join(BASE_DIR, "..", "data", "processed", "financial_data.json")
    
    records = load_and_clean_data(raw_data_dir)
    save_to_json(records, out_json)
    print(f"Data successfully cleaned and saved to {out_json}")
