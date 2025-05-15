
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent  # Points to Stock_AI_Predictor/
sys.path.append(str(project_root))
import pandas as pd
import sqlite3
from Data.Database.db import Database
from Sentiment.alphavantage_api import get_news_sentiment_analysis
from TwitterAPI_Sentiment import TwitterAPI

def load_training_data(stock_id=3, start_date="2024-01-01", end_date="2025-01-01", lookahead_hours=6, window_size=24):
    db = Database()
    twitter_api = TwitterAPI(db)
    clusters = db.get_clusters_by_stock_id(stock_id)
    stock_data = db.get_stock_data_range(stock_id, start_date, end_date)
    stock_data = stock_data.sort_index()

    sentiment_cache = {}
    data_samples = []
    dates = stock_data.index

    for i in range(window_size, len(dates) - lookahead_hours):
        date = dates[i]
        window = stock_data.iloc[i - window_size:i]
        
        print(f"Processing window: {window}")

        if window.isnull().any().any():
            continue  # Skip incomplete windows

        current_price = stock_data.iloc[i]["ClosePrice"]
        future_price = stock_data.iloc[i + lookahead_hours]["ClosePrice"]
        actual_return = (future_price - current_price) / current_price

        # === Get Sentiment (cache daily) ===
        date_key = str(date.date())
        if date_key not in sentiment_cache:
            sentiment_news = get_news_sentiment_analysis("AAPL", date_key)
            sentiment_twitter = twitter_api.get_tweets_sentiment_analysis(ticker_id=stock_id, specific_date=date_key)
            sentiment_cache[date_key] = {
                "impact_score": sentiment_news.get("Predicted Impact Score", 0),
                "news_score": sentiment_news.get("Predicted News Sentiment Score", 0),
                "twitter_score": sentiment_twitter.get("tweets_sentiment_score", 0),
            }

        sentiment = sentiment_cache[date_key]

        # === Pattern Recognition Placeholder ===
        # This should use SVM prediction logic
        cluster_row = clusters.sample(1).iloc[0]

        # === Build Training Sample ===
        data_samples.append({
            "pattern": {
                "probability": cluster_row["ProbabilityScore"],
                "reward_risk_ratio": cluster_row["MaxGain"] / abs(cluster_row["MaxDrawdown"] + 1e-6),
                "max_gain": cluster_row["MaxGain"],
                "max_drawdown": cluster_row["MaxDrawdown"],
            },
            "sentiment": sentiment,
            "price": current_price,
            "actual_return": actual_return
        })

    return data_samples
   
def load_data_from_db(db_path="RL/samples.db", table_name="rl_dataset"):
    """
    Loads data from a SQLite database into a pandas DataFrame.

    Parameters:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to query.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

if __name__ == "__main__":
    # Example usage
    #training_data = load_training_data(stock_id=2, start_date="2024-01-01", end_date="2024-01-05", lookahead_hours=6)
    training_data = load_data_from_db()
    print(training_data)