import numpy as np
import yfinance as yf
import pandas as pd

def get_stock_data_from_yahoo(stock_ID,date, time_frame='1d', period=7):
   
    # to the yahoo finance ticker
    tickers = {
        1: "GC=F",
        2: "BTC-USD",
        3: "AAPL",
        4: "AMZN",
        5: "NVDA",
    }
    stock_symbol = tickers.get(stock_ID, None)
    """
    Retrieves stock data from Yahoo Finance and returns a formatted DataFrame
    similar to the one fetched from your database.

    Parameters:
    - stock_symbol: The stock symbol (e.g., 'AAPL')
    - time_frame: Data interval (e.g., '1d', '1h', '5m')
    - period: Duration to retrieve (e.g., '1mo', '6mo', '1y')
    - stock_ID: Custom Stock ID to include in the DataFrame

    Returns:
    - Pandas DataFrame with columns: 
      ['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 
       'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice']
    """
    start_date = pd.to_datetime(date) - pd.Timedelta(days=period) 
    end_date = pd.to_datetime(date) 
    # Download stock data from Yahoo Finance
    df = yf.download(stock_symbol, interval=time_frame, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No data found for the specified symbol and timeframe.")
    # Convert index from UTC to UTC+3 (Riyadh time), then remove timezone info
    df.index = df.index.tz_convert('Asia/Riyadh').tz_localize(None)
    
    # Reset index to access timestamp
    df.reset_index(inplace=True)


    # Add additional required columns
    df['StockEntryID'] = range(1, len(df) + 1)
    df['StockID'] = stock_ID
    df['StockSymbol'] = stock_symbol
    df['TimeFrame'] = time_frame.replace('m', '')
    df.rename(columns={
        'Open': 'OpenPrice',
        'Close': 'ClosePrice',
        'High': 'HighPrice',
        'Low': 'LowPrice',
        'Date': 'Timestamp'  # some intervals use 'Datetime'
    }, inplace=True)

    # Reorder columns
    df = df[['StockEntryID', 'StockID', 'StockSymbol', 'Datetime' if 'Datetime' in df.columns else 'Timestamp',
             'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice']]
    
    # Ensure Timestamp is datetime and set it as index
    df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    # reverse the order of the DataFrame to match the database format , so tha last value will be the first one
    df = df.iloc[::-1]
    
    

    return df
 
    
if __name__ == "__main__":
    
    data = get_stock_data_from_yahoo(2,date="2025-05-01", time_frame='60m', period=7)
    print(data)
    date_datetime = pd.to_datetime('2025-05-01')
    start_time = date_datetime - pd.Timedelta(hours=70)
    print(f"Start time: {start_time}")
    window = data.loc[(data.index >= start_time) & (data.index <= date_datetime)]
    if window.empty:
            start_time = date_datetime - pd.Timedelta(hours=140)
            window = data.loc[(data.index >= start_time) & (data.index <= date_datetime)]
    window_prices = np.array(window['ClosePrice'])
    # Convert to list if needed
    window_prices = window_prices.flatten()
    print(window_prices)
   
