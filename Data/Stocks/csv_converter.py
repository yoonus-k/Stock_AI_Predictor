import pandas as pd

# Read the CSV (assuming tab-separated)
df = pd.read_csv('Data/Stocks/XAUUSD_M15_201801020000_202504102345.csv', sep='\t')

# Remove < > from column names and select needed columns
df.columns = df.columns.str.replace('[<>]', '', regex=True)
df = df[['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']]

# Rename columns (VOL → Volume, TIME → Time without seconds)
df = df.rename(columns={'DATE': 'Date', 'TIME': 'Time', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOL': 'Volume'})

# Remove seconds from Time (optional)
df['Time'] = df['Time'].str[:5]

# Save as new CSV
df.to_csv('XAUUSDM15.csv', index=False)