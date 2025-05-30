# this is an api interface to get the cot data givin a ticker
import cot_reports as cot
import pandas as pd
# ticker with the cot data name mapping
ticker_cot_mapping = {
    "XAUUSD": "GOLD - COMMODITY EXCHANGE INC.",
    "NASDAQ": "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
    "GBPUSD": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE"
}

# Function to get the COT data for a given ticker
def get_cot_data(ticker):
    if ticker in ticker_cot_mapping:
        cot_name = ticker_cot_mapping[ticker]
        df = cot.cot_all(cot_report_type='legacy_fut', store_txt=False)
        filtered_df = df[df["Market and Exchange Names"] == cot_name]
        # filtering the required columns
        filtered_df = filtered_df[["Market and Exchange Names", "As of Date in Form YYYY-MM-DD", 
                                    "Noncommercial Positions-Long (All)", "Noncommercial Positions-Short (All)", 
                                    "Change in Noncommercial-Long (All)", "Change in Noncommercial-Short (All)",
                                    "% of OI-Noncommercial-Long (All)", "% of OI-Noncommercial-Short (All)",
                                    "Nonreportable Positions-Long (All)", "Nonreportable Positions-Short (All)",
                                    "Change in Nonreportable-Long (All)", "Change in Nonreportable-Short (All)",]]
        # renaming the columns for simple terms like long, short, change in long, change in short
        # for example: "Noncommercial Positions-Long (All)" to "Noncommercial Long"
        filtered_df = filtered_df.rename(columns={
            "Market and Exchange Names": "ticker",
            "As of Date in Form YYYY-MM-DD": "date",
            "Noncommercial Positions-Long (All)": "noncommercial_long",
            "Noncommercial Positions-Short (All)": "noncommercial_short",
            "Change in Noncommercial-Long (All)": "change_noncommercial_long",
            "Change in Noncommercial-Short (All)": "change_noncommercial_short",
            "% of OI-Noncommercial-Long (All)": "per_noncommercial_long",
            "% of OI-Noncommercial-Short (All)": "per_noncommercial_short",
            "Nonreportable Positions-Long (All)": "nonrept_long",
            "Nonreportable Positions-Short (All)": "nonrept_short",
            "Change in Nonreportable-Long (All)": "change_nonrept_long",
            "Change in Nonreportable-Short (All)": "change_nonrept_short"
        })
        # change ticker to the ticker name
        filtered_df["ticker"] = ticker
        
        # sort by date
        filtered_df = filtered_df.sort_values(by="date", ascending=False)
        # drop the index
        filtered_df = filtered_df.reset_index(drop=True)
        # make the date column as datetime
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])
        
        # add the net positions for noncommercial and nonreportable
        filtered_df["net_noncommercial"] = filtered_df["noncommercial_long"] - filtered_df["noncommercial_short"]
        filtered_df["net_nonreportable"] = filtered_df["nonrept_long"] - filtered_df["nonrept_short"]

        # set the date as index
        filtered_df.set_index("date", inplace=True)
        return filtered_df
    else:
        raise ValueError("Ticker not found in COT mapping.")

# Example usage
if __name__ == "__main__":
    ticker = "XAUUSD"  # Example ticker
    try:
        cot_data = get_cot_data(ticker)
        print(cot_data.head())  # Display the first few rows of the COT data
        
    except ValueError as e:
        print(e)