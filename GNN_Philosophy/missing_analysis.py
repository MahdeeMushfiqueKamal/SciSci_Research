import pandas as pd
import numpy as np


df = pd.read_csv('data_raw/Philosophy_2016_Author_Profiles.csv')
total_rows = len(df)
missing_values = df.isnull().sum()
missing_percentage = (missing_values / total_rows) * 100
missing_stats = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Values': missing_values.values,
    'Missing Percentage': missing_percentage.values.round(2)
})

missing_stats = missing_stats.sort_values('Missing Percentage', ascending=False).reset_index(drop=True)
print(missing_stats)