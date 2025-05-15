import os
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import the Database handler if available
try:
    # Get the current working directory
    current_dir = Path(os.getcwd())
    # Navigate to the project root directory if needed
    project_root = current_dir
    if "Stock_AI_Predictor" in str(current_dir):
        while not (project_root / "README.md").exists() and str(project_root) != project_root.root:
            project_root = project_root.parent
    
    import sys
    sys.path.append(str(project_root))
    from Data.Database.db import Database
except ImportError:
    print("Warning: Could not import Database class. Will use direct SQLite connection.")
    Database = None

# API key: 80LCP07NHD21JDYE
# Default URL: https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo

def get_news_sentiment_analysis(ticker, dt, api_key=None, store_to_db=True, stock_id=None):
    """
    Get news sentiment analysis for a specific stock and date.
    
    Args:
        ticker (str): Stock ticker symbol
        dt (str): Date to analyze in YYYY-MM-DD format
        api_key (str): Alpha Vantage API key
        store_to_db (bool): Whether to store the results in the database
        stock_id (int): Stock ID for database storage
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        # Map ticker to Alpha Vantage format if needed
        ticker_mapping = {
            'XAUUSD': 'FOREX:USD',
            'BTCUSD': 'CRYPTO:BTC',
            'AAPL': 'AAPL',
            'AMZN': 'AMZN',
            'NVDA': 'NVDA'
        }
        stock_id_mapping = {
            'XAUUSD': 1,
            'BTCUSD': 2,
            'AAPL': 3,
            'AMZN': 4,
            'NVDA': 5
        }
        alpha_vantage_ticker = ticker_mapping.get(ticker, ticker)
        stock_id = stock_id_mapping.get(ticker, None)
        
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        
        # convert from yyyy-mm-dd hh:mm:ss to yyyy-mm-dd
        db_date = dt.strftime('%Y-%m-%d')
    
        
        # Alpha Vantage requires time window
        time_from = (dt- timedelta(days=1)).strftime('%Y%m%dT%H%M')  # Use the previous day
        time_to = dt.strftime('%Y%m%dT%H%M')  # Use the same day
        
        print(f"Fetching news sentiment for { alpha_vantage_ticker} from {time_from} to {time_to}")
        
        # API endpoint
        url = 'https://www.alphavantage.co/query'
        
        # Parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': alpha_vantage_ticker,
            'time_from': time_from,
            'time_to': time_to,
            'limit': 200,
            'apikey': api_key or '80LCP07NHD21JDYE' # Default API key
        }
        
        # Make API request
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'feed' not in data or not data['feed']:
            return {
                'Predicted News Sentiment Score': 0,
                'Predicted Impact Score': 0,
                'News Count': 0,
                'Bullish Ratio': 0,
                'Summary of the News': 'No news articles found for this date',
                'Top Topics': [],
                'Most Relevant Article': None
            }
        
        # Initialize metrics
        sentiment_scores = []
        impact_scores = []
        relevance_scores = []
        bullish_count = 0
        bearish_count = 0
        topics = {}
        most_relevant_article = None
        max_relevance = 0
        # Store article data for database
        articles_to_store = []
       
        for item in data['feed']:
             # Check if the article contains ticker sentiments for our stock
            ticker_sentiments = item.get('ticker_sentiment', [])
            
            # Skip this article if it doesn't contain ticker sentiments
            max_relevance_score = 0
            most_relevant_ticker = None
            most_relevent_stock_sentiment_score = 0
            most_relevent_stock_sentiment_label = 'Neutral'
            
            # First pass: find the most relevant ticker in this article
            for ts in ticker_sentiments:
                relevance_score = float(ts.get('relevance_score', 0))
                if relevance_score > max_relevance_score:
                    max_relevance_score = relevance_score
                    most_relevant_ticker = ts.get('ticker')
                    most_relevent_stock_sentiment_score = float(ts.get('ticker_sentiment_score', 0))
                    most_relevent_stock_sentiment_label = ts.get('ticker_sentiment_label', 'Neutral')
            
                                
            # Skip this article if:
            # 1. Our target ticker is not in the ticker_sentiments list, or
            # 2. Our target ticker is not the most relevant one (has the highest relevance score)
            if  most_relevant_ticker != alpha_vantage_ticker:
                continue
            
            # Format the date from 'YYYYMMDDTHHMMSS' to 'YYYY-MM-DD HH:MM:SS'
            time_published = item.get('time_published', '')
            formatted_date = ''
            if time_published and len(time_published) >= 15:
                try:
                    date_obj = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    formatted_date = time_published  # Use original if parsing fails
            
            article_data = {
                'date': formatted_date,
                'authors': item.get('authors', []),
                'source_domain': item.get('source_domain', ''),
                'source_name': item.get('source', ''),
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'url': item.get('url', ''),
                'event_type': item.get('category_within_source', 'News'),  # Add event_type from category_within_source
                'topics': json.dumps(item.get('topics', [])),
                #'ticker_sentiment': json.dumps(item.get('ticker_sentiment', [])),
                'overall_sentiment_label': item.get('overall_sentiment_label', 'Neutral'),  # Default value
                'overall_sentiment_score': item.get('overall_sentiment_score', 0.0),  # Default value
                'fetch_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Use standardized format
                'most_relevant_stock_id': stock_id,  # Add most relevant stock ID
                'most_relevant_stock_sentiment_score': most_relevent_stock_sentiment_score,
                'most_relevant_stock_sentiment_label': most_relevent_stock_sentiment_label,
            }
            
            # Add to the list of articles to store
            articles_to_store.append(article_data)
          
        # if there is no relevant news, return early
        if len(articles_to_store) == 0:
            most_relevant_article = {
                'title': 'No relevant news found',
                'summary': 'No relevant news found',
                'url': '',
                'source': '',
                'time_published': '',
                'relevance_score': 0,
                'sentiment_score': 0,
            }
        
        for article in articles_to_store:
            # get the sentiment and relevance scores
            over_all_sentiment_score = float(article['overall_sentiment_score'])
            most_relevent_stock_sentiment_score = float(article['most_relevant_stock_sentiment_score'])
            
            sentiment_scores.append(over_all_sentiment_score)
            relevance_scores.append(most_relevent_stock_sentiment_score )
            impact_scores.append(over_all_sentiment_score * most_relevent_stock_sentiment_score )  # Weighted impact
            
            # Count bullish and bearish articles (sentiment > 0.15)
            if  over_all_sentiment_score > 0.15:
                bullish_count += 1
            elif over_all_sentiment_score < -0.15:
                bearish_count += 1
            
            # Track most relevant article
            if most_relevent_stock_sentiment_score > max_relevance:
                most_relevant_article = {
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'time_published': item.get('time_published', ''),
                    'relevance_score': most_relevent_stock_sentiment_score,
                    'sentiment_score': over_all_sentiment_score,
                }
                max_relevance = most_relevent_stock_sentiment_score
        
        # Analyze topics
        
        for topic in item.get('topics', []):
            topic_name = topic.get('topic', '')
            topic_relevance = float(topic.get('relevance_score', 0))
            if topic_name:
                topics[topic_name] = topics.get(topic_name, 0) + topic_relevance
                
          # Store in database if requested
        if store_to_db and stock_id is not None:
            try:
                # Store the articles and their relations in the database
                if Database is not None:
                    db = Database()
                    db.store_articles_in_database(articles_to_store, stock_id, db_date)
                    print(f"Successfully stored {len(articles_to_store)} articles in the database")
                else:
                    print("Database class not available, articles not stored")
            except Exception as e:
                print(f"Error storing articles in database: {e}")
        
        if not sentiment_scores:
            return {
                'Predicted News Sentiment Score': 0,
                'Predicted Impact Score': 0,
                'News Count': 0,
                'Bullish Ratio': 0,
                'Bearish Ratio': 0,
                'Summary of the News': 'No relevant news found for this stock',
                'Top Topics': [],
                'Most Relevant Article': most_relevant_article
            }
        
        # Calculate metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) # Average sentiment
        avg_impact = sum(impact_scores) / sum(relevance_scores) if sum(relevance_scores) > 0 else 0 # Average impact score by relevance
        bullish_ratio = bullish_count / len(sentiment_scores) * 100 if sentiment_scores else 0 # Bullish ratio
        bearish_ratio = bearish_count / len(sentiment_scores) * 100 if sentiment_scores else 0 # Bearish ratio
        
        # Get top 3 topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
        top_topics = [topic[0] for topic in sorted_topics]
        
        
        return {
            'Predicted News Sentiment Score': round(avg_sentiment, 2),
            'Predicted Impact Score': round(avg_impact, 2),
            'News Count': len(sentiment_scores),
            'Bullish Ratio': round(bullish_ratio, 1),
            'Bearish Ratio': round(bearish_ratio, 1),
            'Summary of the News': most_relevant_article['summary'] if most_relevant_article else 'No relevant article found',
            'Top Topics': top_topics,
            'Most Relevant Article': most_relevant_article
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'error': str(e),
            'Predicted News Sentiment Score': 0,
            'Predicted Impact Score': 0,
            'News Count': 0,
            'Bullish Ratio': 0,
            'Bearish Ratio': 0,
            'Summary of the News': 'Error processing news data',
            'Top Topics': [],
            'Most Relevant Article': most_relevant_article
        }


# main function
if __name__ == '__main__':
    # Example usage:
    api_key = '80LCP07NHD21JDYE'
    stock_ticker = 'AAPL'  # For gold you would use 'GC'
    date_str = '2025-05-09 16:00:00'  # Matches one of the articles in your sample

    result = get_news_sentiment_analysis(stock_ticker, date_str, api_key)
    print(result)
    # format the output
    print(f"Predicted News Sentiment Score: {result['Predicted News Sentiment Score']}")
    print(f"Predicted Impact Score: {result['Predicted Impact Score']}")
    print(f"News Count: {result['News Count']}")
    print(f"Bullish Ratio: {result['Bullish Ratio']}%")
    print(f"Bearish Ratio: {result['Bearish Ratio']}%")
    print(f"Summary of the News: {result['Summary of the News']}")
    print(f"Top Topics: {', '.join(result['Top Topics'])}")

    print(f"Most Relevant Article Title: {result['Most Relevant Article']['title']}")
    print(f"Most Relevant Article summary: {result['Most Relevant Article']['summary']}")
    print(f"Most Relevant Article URL: {result['Most Relevant Article']['url']}")
