# Import necessary libraries
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



from dotenv import load_dotenv
from Data.db_cloud import Database  # Adjust the import based on your project structure
# Initialize database connection
db = Database()

# Load environment variables (for API keys)
load_dotenv()


class TwitterAPI:
    """
    Class to interact with Twitter API for fetching tweets based on search queries.
    
    Attributes:
        api_key (str): Twitter API key.
        base_url (str): Base URL for Twitter API.
    """
    def __init__(self,db=None, api_key='6da37e966bbc4ec097848830c771b7c0', base_url="https://api.twitterapi.io/twitter/tweet/advanced_search"):
        self.api_key = api_key
        self.base_url = base_url
        self.db = db if db else Database()
        # Define ticker mapping
        self.ticker_mapping = {
            1: "XAUUSD",
            2: "BTCUSD",
            3: "AAPL",
            4: "AMZN",
            5: "NVDA"
        }
                # Dictionary of search term weights
        self.SEARCH_TERM_WEIGHTS = {
            "AAPL OR Apple -from:Apple": 2.5,  # 2.5x weight
            "Tim Cook -from:Apple": 2,        # 2x weight
            "Apple earnings OR (AAPL earnings)": 1.7, # 1.7x weight
        }
        self.DEFAULT_SEARCH_TERM_WEIGHT = 0.5
        self.min_tweet_count=10
    # Define search queries based on ticker ID
    def get_search_queries_by_ticker(self,ticker_id):
        """
        Get appropriate search queries based on ticker ID
        
        Args:
            ticker_id (int): ID of the ticker
            
        Returns:
            list: List of search queries
        """
        if ticker_id not in self.ticker_mapping:
            print(f"Error: Ticker ID {ticker_id} not found in mapping")
            return []
        
        ticker = self.ticker_mapping[ticker_id]
        # Gold
        if ticker == "XAUUSD":  
            return [
                "XAUUSD OR Gold price min_retweets:20",
                "Gold trading OR Gold market min_retweets:20",
                "Gold investment min_retweets:20",
            ]
        # Bitcoin
        elif ticker == "BTCUSD":  
            return [
                "BTCUSD OR Bitcoin price min_retweets:20",
                "Bitcoin trading OR Bitcoin market min_retweets:20",
            ]
        # Apple
        elif ticker == "AAPL":  
            return [
                "AAPL OR $AAPL -from:Apple min_retweets:20",
                "Tim Cook -from:Apple min_retweets:20",
            ]
        elif ticker == "AMZN":  # Amazon
            return [
                "AMZN OR Amazon -from:Amazon min_retweets:20",
                "Jeff Bezos OR Andy Jassy -from:Amazon min_retweets:20",
            ]
        elif ticker == "NVDA":  # Nvidia
            return [
                "NVDA OR Nvidia -from:Nvidia min_retweets:20",
                "Jensen Huang -from:Nvidia min_retweets:20",
            ]


        # elif ticker == "MSFT":  # Microsoft
        #     return [
        #         "MSFT OR Microsoft -from:Microsoft",
        #         "Satya Nadella -from:Microsoft",
        #         "Windows min_retweets:10",
        #         "Xbox OR (Microsoft Gaming) min_likes:25",
        #         "Surface OR (Surface Book) OR (Surface Pro)",
        #         "Microsoft 365 OR Office365",
        #         "Azure OR (Microsoft Cloud)",
        #         "Microsoft earnings OR (MSFT earnings)",
        #         "(Microsoft stock) OR (MSFT stock) OR (Microsoft shares)",
        #         "Microsoft AI OR (Microsoft Copilot)",
        #         "Microsoft Teams",
        #         "GitHub"
        #     ]
        # elif ticker == "GOOGL":  # Google
        #     return [
        #         "GOOGL OR Google -from:Google",
        #         "Sundar Pichai -from:Google",
        #         "Android min_retweets:10",
        #         "Google I/O OR (Google event) min_likes:25",
        #         "Pixel OR (Google Pixel) OR (Pixel Pro)",
        #         "Chrome OR ChromeOS",
        #         "Google Cloud",
        #         "Google earnings OR (GOOGL earnings)",
        #         "(Google stock) OR (GOOGL stock) OR (Google shares)",
        #         "Google AI OR (Google Gemini)",
        #         "YouTube"
        #     ]
        # Add more ticker query sets as needed
        else:
            # Generic queries for any ticker
            return [
                f"{ticker}",
                f"${ticker}",
                f"{ticker} price",
                f"{ticker} trading",
                f"{ticker} stock",
                f"{ticker} market",
                f"{ticker} analysis",
                f"{ticker} forecast",
                f"{ticker} earnings",
                f"{ticker} news"
            ]

    def check_tweets_in_db_for_date(self,start_date, end_date, ticker_id=0):
        """
        Check if tweets exist in DB for the specified date range and ticker
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            ticker_id (int, optional): ID of the ticker
            
        Returns:
            tuple: (bool, int) - (exists, count)
        """
        try:
            # Convert to datetime objects
            start_dt = self.format_date_for_query(start_date)
            end_dt = self.format_date_for_query(end_date)

            
            # Add one day to end_date to include the full day
            end_dt = end_dt + timedelta(days=1)
            
            # Format for database query
            start_str = start_dt.strftime('%Y-%m-%d')
            end_str = end_dt.strftime('%Y-%m-%d')
            
            # Query database
            tweets_df = self.db.get_tweets_count_by_date_and_ticker(
                start_date=start_str,
                end_date=end_str,
                ticker_id=ticker_id
            )
            
            # Calculate total count
            tweet_count = tweets_df['tweet_count'].sum() if not tweets_df.empty else 0
            
            # Check if tweets exist
            exists = tweet_count > 0
            
            ticker_name = self.ticker_mapping.get(ticker_id, f"ticker {ticker_id}") if ticker_id in self.ticker_mapping else ""
            print(f"Found [{tweet_count}] tweets in database for period [{start_date}] to [{end_date}]" +
                (f" for ticker_id: {ticker_id} and ticker_name: {ticker_name}" if ticker_name else ""))
            
            return exists, tweet_count
            
        except Exception as e:
            print(f"Error checking tweets in database: {e}")
            return False, 0
        
    from datetime import datetime, timedelta

    def format_date_for_query(self,date_input):
        """Ensure date_input is parsed correctly even if it includes time."""
        if isinstance(date_input, datetime):
            # Already a datetime object, no parsing needed
            return date_input
        try:
            # Try parsing full datetime first
            return datetime.strptime(date_input, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If fails, try parsing just the date
            return datetime.strptime(date_input, '%Y-%m-%d')


    def fetch_tweets(self,query, date_from=None, date_to=None, query_type="Latest", max_tweets=1000):
        """
        Fetch tweets using the TwitterAPI.io Advanced Search endpoint
        
        Args:
            query (str): Twitter advanced search query
            date_from (str, optional): Start date in YYYY-MM-DD format
            date_to (str, optional): End date in YYYY-MM-DD format
            query_type (str): "Latest" or "Top" tweets
            max_tweets (int): Maximum number of tweets to retrieve
            
        Returns:
            list: List of tweet objects
        """
        all_tweets = []
        cursor = ""
        page_count = 0
        max_pages = (max_tweets // 100) + 1  # API typically returns ~100 tweets per page
        
        # Modify query to include date range if provided
        if date_from and date_to:
            print(f"Fetching tweets from {date_from} to {date_to}")
            
            # Adapt parsing here
            date_from_dt = self.format_date_for_query(date_from) - timedelta(days=1)
            date_to_dt = self.format_date_for_query(date_to)
            
            date_from_str = date_from_dt.strftime('%Y-%m-%d_%H:%M:%S_UTC')
            date_to_str = date_to_dt.strftime('%Y-%m-%d_%H:%M:%S_UTC')
            
            # Add to query
            query = f"{query} since:{date_from_str} until:{date_to_str}"

            
        print(f"Fetching tweets for query: {query}")
        
        while len(all_tweets) < max_tweets and page_count < max_pages:
            # Prepare request parameters
            params = {
                "query": query,
                "queryType": query_type
            }
            
            # Add cursor for pagination if not first page
            if cursor:
                params["cursor"] = cursor
            
            # Prepare headers with API key
            
            headers = {
                "X-API-Key": f"{self.api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                # Make API request
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers
                )
                
                # Check for successful response
                response.raise_for_status() # If the response is not 200, this will raise an HTTPError
                
                # Parse response
                data = response.json() 
                
                # Add tweets to our collection
                new_tweets = data.get("tweets", [])
                all_tweets.extend(new_tweets)
                
                # Update cursor for next page
                cursor = data.get("next_cursor", "")
                has_next_page = data.get("has_next_page", False)
                
                page_count += 1
                
                print(f"Page {page_count}: Retrieved {len(new_tweets)} tweets. Total: {len(all_tweets)}")
                
                # If no more pages or we have enough tweets, break
                if not has_next_page or not cursor:
                    print(f"No more pages for {query} or cursor is empty. Stopping fetch.")
                    break
                    
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching tweets: {e}")
                break
        
        return all_tweets

    def process_tweet(self,tweet, search_term, ticker_id=0):
        """
        Process a single tweet and extract relevant fields without sentiment analysis
        
        Args:
            tweet (dict): Tweet object from the API
            search_term (str): The search term that found this tweet
            ticker_id (int, optional): ID of the ticker
            
        Returns:
            dict: Processed tweet data
        """
        # Current timestamp
        current_time = datetime.now().isoformat()
        
        # Convert tweet creation time to ISO format
        created_at = tweet.get('createdAt', '')
        if created_at:
            created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").isoformat()
            
        # Extract empty placeholders for sentiment fields (to be filled later)
        sentiment_label = ""
        sentiment_score = 0.0
        sentiment_magnitude = 0.0
        weighted_sentiment = 0.0
        
        # Return processed tweet data
        return {
            'tweet_id': tweet.get('id', ''),
            'tweet_text': tweet.get('text', ''),
            'created_at': created_at,
            'retweet_count': tweet.get('retweetCount', 0),
            'reply_count': tweet.get('replyCount', 0),
            'like_count': tweet.get('likeCount', 0),
            'quote_count': tweet.get('quoteCount', 0),
            'bookmark_count': tweet.get('bookmarkCount', 0),
            'lang': tweet.get('lang', ''),
            'is_reply': tweet.get('isReply', False),
            'is_quote': bool(tweet.get('quoted_tweet')),
            'is_retweet': bool(tweet.get('retweeted_tweet')),
            'url': tweet.get('url', ''),
            'search_term': search_term,
            'author_username': tweet.get('author', {}).get('userName', ''),
            'author_name': tweet.get('author', {}).get('name', ''),
            'author_verified': False,  # TwitterAPI.io doesn't have this field, only isBlueVerified
            'author_blue_verified': tweet.get('author', {}).get('isBlueVerified', False),
            'author_followers': tweet.get('author', {}).get('followers', 0),
            'author_following': tweet.get('author', {}).get('following', 0),
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'sentiment_magnitude': sentiment_magnitude,
            'weighted_sentiment': weighted_sentiment,
            'collected_at': current_time,
            'ticker_id': ticker_id,
        }

    def store_tweets_in_db(self,tweets):
        """
        Store processed tweets in the database
        
        Args:
            tweets (list): List of processed tweet dictionaries
            
        Returns:
            int: Number of tweets stored
        """
        stored_count = 0
        
        for tweet in tweets:
            try:
                self.db.store_tweets(
                    ticker_id=tweet['ticker_id'],
                    tweet_id=tweet['tweet_id'],
                    tweet_text=tweet['tweet_text'],
                    created_at=tweet['created_at'],
                    retweet_count=tweet['retweet_count'],
                    reply_count=tweet['reply_count'],
                    like_count=tweet['like_count'],
                    quote_count=tweet['quote_count'],
                    bookmark_count=tweet['bookmark_count'],
                    lang=tweet['lang'],
                    is_reply=tweet['is_reply'],
                    is_quote=tweet['is_quote'],
                    is_retweet=tweet['is_retweet'],
                    url=tweet['url'],
                    search_term=tweet['search_term'],
                    author_username=tweet['author_username'],
                    author_name=tweet['author_name'],
                    author_verified=tweet['author_verified'],
                    author_blue_verified=tweet['author_blue_verified'],
                    author_followers=tweet['author_followers'],
                    author_following=tweet['author_following'],
                    sentiment_label=tweet['sentiment_label'],
                    sentiment_score=tweet['sentiment_score'],
                    sentiment_magnitude=tweet['sentiment_magnitude'],
                    weighted_sentiment=tweet['weighted_sentiment'],
                    collected_at=tweet['collected_at'],
                )
                stored_count += 1
                
            except Exception as e:
                print(f"Error storing tweet {tweet['tweet_id']}: {e}")
        
        return stored_count

    def fetch_and_store_tweets_for_date_range(self,start_date, end_date, ticker_id=None, tweets_per_query=500, language="en"):
        """
        Fetch tweets for a specific date range and ticker, then store in DB
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            ticker_id (int, optional): ID of the ticker
            tweets_per_query (int): Number of tweets per query
            language (str): Language filter
            
        Returns:
            int: Total number of tweets stored
        """
        total_stored = 0
        
        # Get search queries based on ticker ID
        search_queries = self.get_search_queries_by_ticker(ticker_id) if ticker_id else APPLE_SEARCH_QUERIES
        
        if not search_queries:
            print(f"No search queries found for ticker ID {ticker_id}")
            return 0
        
        print(f"\n======= Fetching tweets from {start_date} to {end_date} =======")
        
        # Process each search query
        for base_query in search_queries:
            # Add language filter
            query = f"{base_query} lang:{language}"
            print(f"Processing query: {query}")
            
            # Fetch tweets for this query
            tweets = self.fetch_tweets(
                query=query,
                date_from=start_date,
                date_to=end_date,
                query_type="Latest",
                max_tweets=tweets_per_query
            )
            
            if tweets:
                # Process tweets
                processed_tweets = [self.process_tweet(tweet, base_query, ticker_id) for tweet in tweets]
                
                # Store in database
                stored_count = self.store_tweets_in_db(processed_tweets)
                total_stored += stored_count
                
                print(f"Stored {stored_count} tweets for query '{base_query}' for period {start_date} to {end_date}")
            else:
                print(f"No tweets found for query '{base_query}' for period {start_date} to {end_date}")
        
        print(f"### Tweet fetching and storage completed. Total tweets stored: {total_stored} ###")
        return total_stored

    def fetch_tweets_if_needed(self,specific_date=None, start_date=None, end_date=None, ticker_id=None, min_tweet_count=100):
        """
        Check if tweets exist for the given date parameters and fetch if needed
        
        Args:
            specific_date (str, optional): A single date to analyze (YYYY-MM-DD)
            start_date (str, optional): Start date of range (YYYY-MM-DD)
            end_date (str, optional): End date of range (YYYY-MM-DD)
            ticker_id (int, optional): ID of the ticker
            min_tweet_count (int): Minimum number of tweets needed
            
        Returns:
            tuple: (start_date, end_date, needs_analysis)
        """
        # Process date parameters
        if specific_date:
            start_date = specific_date
            end_date = specific_date
        elif start_date and end_date:
            # Use provided date range
            pass
        else:
            # Default to yesterday if no date parameters
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = yesterday
            end_date = yesterday
            
        # Check if we have enough tweets in the database
        tweets_exist, tweet_count = self.check_tweets_in_db_for_date(start_date, end_date, ticker_id)
        
        if not tweets_exist or tweet_count < self.min_tweet_count:
            print(f"Insufficient tweets in database for the requested date range.\n[Tweets_Count={tweet_count}], [Minimum_Tweets_Count={self.min_tweet_count}]\n=========Fetching from API...=========")
            self.fetch_and_store_tweets_for_date_range(start_date, end_date, ticker_id)
            return start_date, end_date, True
        else:
            print(f"Sufficient tweets found in database. Proceeding with analysis...")
            return start_date, end_date, False

    # Define Apple search queries (default if no ticker specified)
    APPLE_SEARCH_QUERIES = [
        "AAPL OR Apple -from:Apple",
        "Tim Cook -from:Apple",
        "iPhone min_retweets:10",
        "WWDC OR (Apple event) min_likes:25",
        "MacBook OR Macbook OR (Mac Pro) OR iMac",
        "iPad OR iPadOS",
        "iOS OR iPadOS OR macOS",
        "Apple earnings OR (AAPL earnings)",
        "(Apple stock) OR (AAPL stock) OR (Apple shares)",
        "Apple AI OR (Apple intelligence)",
        "Apple Vision Pro",
        "Apple Watch",
        "AirPods OR (Apple headphones)"
    ]




    # Set pandas display options
    pd.set_option('display.max_colwidth', 100)

    

    # Function to preprocess tweet text
    def preprocess(self,text):
        """
        Preprocess tweet text by handling mentions and links
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    # Function to load tweets from database based on date range and ticker
    def load_tweets_from_db(self,start_date=None, end_date=None, ticker_id=0):
        """
        Load tweets from SQLite database based on date range and ticker
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            ticker_id (int, optional): ID of the ticker
            
        Returns:
            pd.DataFrame: DataFrame with loaded tweets
        """
        try:
            # Convert to datetime objects if provided
            start_dt = None
            end_dt = None
            
            start_dt = self.format_date_for_query(start_date) - timedelta(days=1) if start_date else None
            end_dt = self.format_date_for_query(end_date) if end_date else None
            
            
            # Format for database query
            start_str = start_dt.strftime('%Y-%m-%d') if start_dt else None
            end_str = end_dt.strftime('%Y-%m-%d') if end_dt else None
            print(f"Loading tweets from database for period [{start_str}] to [{end_str}]")
            # Get tweets from the database
            tweets = self.db.get_tweets_by_date_and_ticker(
                start_date=start_str,
                end_date=end_str,
                ticker_id=ticker_id
            )
            
            # Convert to DataFrame
            df = tweets
            
            filter_description = ""
            if start_date and end_date:
                filter_description += f"date range {start_date} to {end_date}"
            elif start_date:
                filter_description += f"from {start_date}"
            elif end_date:
                filter_description += f"until {end_date}"
                
            if ticker_id:
                ticker_name = self.ticker_mapping.get(ticker_id, f"ticker_id {ticker_id}")
                if filter_description:
                    filter_description += f" with ticker {ticker_name}"
                else:
                    filter_description += f"ticker {ticker_name}"
                    
            print(f"Loaded [{len(df)}] tweets from database" + (f" for {filter_description}" if filter_description else ""))
            
            # Convert date columns to datetime if they exist
            for col in ['created_at', 'collected_at']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Apply preprocessing to tweet text
            if 'tweet_text' in df.columns:
                df['preprocessed_text'] = df['tweet_text'].apply(self.preprocess)
            
            return df
                    
        except Exception as e:
            print(f"Error loading tweets from database: {e}")
            return pd.DataFrame()

    # Setup for sentiment analysis with Twitter-RoBERTa
    def setup_twitter_roberta(self):
        # Load the Twitter-RoBERTa tokenizer and model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Move model to GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using NVIDIA GPU.")
            else:
                device = torch.device("cpu")
                print("Using CPU.")
            model = model.to(device)
            return tokenizer, model
        except Exception as e:
            print(f"Error loading Twitter-RoBERTa model: {e}")
            return None, None

    # Function to analyze sentiment with Twitter-RoBERTa
    def analyze_sentiment_roberta(self,text, tokenizer, model, max_length=512):
        if not text or not isinstance(text, str) or not tokenizer or not model:
            return "neutral", 0.0, 0.0
        
        try:
            # Preprocess the text
            preprocessed_text = self.preprocess(text)
            
            # Encode the text
            inputs = tokenizer(preprocessed_text, return_tensors="pt", max_length=max_length, 
                            truncation=True, padding=True)
            
            # Move inputs to GPU if available        
            if torch.cuda.is_available():
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
                
            # Get model output
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get prediction (0: negative, 1: neutral, 2: positive)
            # Note: The Twitter-RoBERTa model has labels ordered as [negative, neutral, positive]
            prediction = predictions[0].tolist()
            sentiment_id = np.argmax(prediction)  
            
            labels = ["negative", "neutral", "positive"]
            sentiment_label = labels[sentiment_id]
            
            # Get confidence score (highest probability)
            confidence = max(prediction)
            
            # Convert to score between -1 and 1
            if sentiment_id == 0:  # negative
                score = -prediction[0]
            elif sentiment_id == 2:  # positive
                score = prediction[2]
            else:  # neutral
                score = 0.0
                
            return sentiment_label, float(score), float(confidence)
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "neutral", 0.0, 0.0

    # Function to add sentiment analysis to DataFrame
    def add_sentiment_to_df(self,df, tokenizer, model):
        # Ensure required columns exist
        required_cols = ['tweet_text']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in DataFrame")
                return df
        
        # Add new columns if they don't exist
        for col in ['sentiment_label', 'sentiment_score', 'sentiment_confidence', 'weighted_sentiment', 'sentiment_value']:
            if col not in df.columns:
                df[col] = None
        
        # Process each row
        for idx, row in df.iterrows():
            # Get tweet text - use preprocessed text if available
            text = row.get('preprocessed_text', row['tweet_text'])
            
            # Analyze sentiment
            sentiment_label, sentiment_score, confidence = self.analyze_sentiment_roberta(text, tokenizer, model)
            
            # Convert sentiment_label to numerical value (-1, 0, 1)
            sentiment_value = -1 if sentiment_label == "negative" else (1 if sentiment_label == "positive" else 0)
            
            # Update DataFrame
            df.at[idx, 'sentiment_label'] = sentiment_label
            df.at[idx, 'sentiment_score'] = sentiment_score
            df.at[idx, 'sentiment_confidence'] = confidence
            df.at[idx, 'sentiment_value'] = sentiment_value
            
            # Calculate basic weighted sentiment (Ss * Sa)
            df.at[idx, 'weighted_sentiment'] = sentiment_score * confidence
        
        return df

    # Function to calculate engagement-weighted sentiment
    def calculate_engagement_weighted_sentiment(self,df, a=0.3, b=0.4, c=0.2, d=0.1, e=0.8, ticker_weight=1.5):
        # Ensure required columns exist
        required_cols = ['retweet_count', 'like_count', 'reply_count', 'author_followers', 'search_term']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found. Using placeholder values.")
                if col == 'retweet_count':
                    df['retweet_count'] = 0
                elif col == 'like_count':
                    df['like_count'] = 0
                elif col == 'reply_count':
                    df['reply_count'] = 0
                elif col == 'author_followers':
                    df['author_followers'] = 0
                elif col == 'search_term':
                    df['search_term'] = ""
        
        # Add new columns
        df['engagement_score'] = 0.0
        df['final_weighted_sentiment'] = 0.0
        
        # Normalize follower counts (to avoid extreme values)
        # Use log scale with a small constant to handle zeros
        if 'author_followers' in df.columns and df['author_followers'].max() > 0:
            df['normalized_followers'] = np.log1p(df['author_followers']) / np.log1p(df['author_followers'].max())
        else:
            df['normalized_followers'] = 0
        
        # Process each row
        for idx, row in df.iterrows():
            # Skip rows with no sentiment data
            if pd.isna(row['sentiment_score']) or pd.isna(row['sentiment_confidence']):
                continue
            
            # Get engagement metrics
            tr = row['retweet_count'] if 'retweet_count' in df.columns else 0
            ti = row['like_count'] if 'like_count' in df.columns else 0
            tc = row['reply_count'] if 'reply_count' in df.columns else 0
            tf = row['normalized_followers'] 
            
            # Calculate engagement score
            engagement = (a * tr + b * ti + c * tc + d * tf)
            
            # User influence factor (using verified status and blue verification as a proxy)
            ui = 1.0  # base influence
            if 'author_verified' in df.columns and row['author_verified']:
                ui += 0.3  # bonus for legacy verified accounts
            if 'author_blue_verified' in df.columns and row['author_blue_verified']:
                ui += 0.1  # smaller bonus for blue verified accounts
            
            # Apply user influence with hyperparameter e
            user_factor = ui * e
            
            # Ticker detection logic
            ticker_factor = 1.0
            
            # Apply ticker weight if ticker_id is present and valid
            if 'ticker_id' in df.columns and row['ticker_id'] > 0:
                ticker_factor = ticker_weight
            
            # Add search term weighting
            search_term_factor = self.DEFAULT_SEARCH_TERM_WEIGHT  # Start with default weight
            if 'search_term' in df.columns and row['search_term']:
                # Apply custom weight if the search term matches one in our dictionary
                if row['search_term'] in self.SEARCH_TERM_WEIGHTS:
                    search_term_factor = self.SEARCH_TERM_WEIGHTS[row['search_term']]
                    
            # Calculate final weighted sentiment
            base_sentiment = row['sentiment_score'] * row['sentiment_confidence']
            final_sentiment = base_sentiment * (1 + engagement) * user_factor * ticker_factor * search_term_factor
            
            # Update DataFrame
            df.at[idx, 'engagement_score'] = engagement
            df.at[idx, 'user_influence'] = user_factor
            df.at[idx, 'ticker_factor'] = ticker_factor
            df.at[idx, 'search_term_factor'] = search_term_factor  
            df.at[idx, 'final_weighted_sentiment'] = final_sentiment
            
        # Normalize all final sentiment values to range between -1 and 1
        if not df['final_weighted_sentiment'].empty:
            max_abs_sentiment = df['final_weighted_sentiment'].abs().max()
            if max_abs_sentiment > 0:  # Avoid division by zero
                df['final_weighted_sentiment'] = df['final_weighted_sentiment'] / max_abs_sentiment
        
        # Alternative normalization using sigmoid function
        # df['final_weighted_sentiment'] = 2 * (1 / (1 + np.exp(-0.001 * df['final_weighted_sentiment']))) - 1
        return df

    # Function to update tweets in the database with sentiment scores
    def update_tweets_in_db(self,df):
        updated_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Get the tweet_id
                tweet_id = row['tweet_id']
                
                # Update the tweet in the database
                self.db.update_tweet_sentiment(
                    tweet_id=tweet_id,
                    sentiment_label=row.get('sentiment_label', ''),
                    sentiment_score=float(row.get('sentiment_score', 0.0)),
                    sentiment_magnitude=float(row.get('sentiment_confidence', 0.0)),  # Using confidence as magnitude
                    weighted_sentiment=float(row.get('final_weighted_sentiment', 0.0))
                )
                
                updated_count += 1
                
            except Exception as e:
                print(f"Error updating tweet {row.get('tweet_id', 'unknown')}: {e}")
        
        print(f"Updated {updated_count} tweets in database")
        return updated_count

    # Function to visualize sentiment analysis results - with two visualizations only
    def visualize_sentiment_analysis(self,df, ticker_id=None):
        # Get ticker symbol if ticker_id is provided
        ticker_symbol = None
        if ticker_id is not None and ticker_id in self.ticker_mapping:
            ticker_symbol = self.ticker_mapping[ticker_id]
        
        # Set title suffix based on ticker
        title_suffix = f" for {ticker_symbol}" if ticker_symbol else ""
        
        # Set up figure with 1 row, 2 columns for the two remaining visualizations
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Sentiment distribution pie chart
        sentiment_counts = df['sentiment_label'].value_counts()
        
        # Define color mapping for sentiment labels
        color_map = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}
        
        # Get colors in the same order as sentiment_counts index
        colors = [color_map[label] for label in sentiment_counts.index]
        
        axes[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                    colors=colors, textprops={'fontsize': 12})
        axes[0].set_title(f'Tweet Sentiment Distribution{title_suffix}', fontsize=14)
        
        # 2. Histogram of sentiment scores
        sns.histplot(df['sentiment_score'], bins=20, ax=axes[1])
        axes[1].set_title(f'Distribution of Sentiment Scores{title_suffix}', fontsize=14)
        axes[1].set_xlabel('Sentiment Score (-1 to 1)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        plt.show()

    def analyze_tweets_for_date_range(self,specific_date=None, start_date=None, end_date=None, ticker_id=None, min_tweet_count=100):
        # from Data.collector import fetch_tweets_if_needed
        
        # Fetch tweets if needed and get date range for analysis
        start_date, end_date, needs_fetch = self.fetch_tweets_if_needed(
            specific_date=specific_date, 
            start_date=start_date, 
            end_date=end_date, 
            ticker_id=ticker_id, 
            min_tweet_count=self.min_tweet_count
        )
        
        # Load tweets from database
        df = self.load_tweets_from_db(start_date, end_date, ticker_id)
        
        # Check if DataFrame is not empty
        if not df.empty:
            # Check if we have enough tweets
            if len(df) < self.min_tweet_count:
                print(f"Insufficient tweets in database for the requested date range.")
                return df
            
            # Setup Twitter-RoBERTa model
            tokenizer, model = self.setup_twitter_roberta()
            
            # Check if tokenizer and model are loaded
            if tokenizer and model:
                # Add sentiment scores to DataFrame
                df = self.add_sentiment_to_df(df, tokenizer, model)
                
                # Calculate engagement-weighted sentiment
                df = self.calculate_engagement_weighted_sentiment(
                    df,
                    a=0.3,  # weight for retweets
                    b=0.4,  # weight for likes
                    c=0.2,  # weight for replies
                    d=0.1,  # weight for followers
                    e=0.8,  # weight for user influence
                    ticker_weight=1.5  # extra weight for ticker symbols
                )
                
                # Update tweets in database if we fetched new data
                if needs_fetch:
                    self.update_tweets_in_db(df)
                
                # Display sentiment statistics
                print("\nSentiment Distribution:")
                print(df['sentiment_label'].value_counts())
                
                print("\nSentiment Score Statistics:")
                print(df['sentiment_score'].describe())
                
                print("\nWeighted Sentiment Statistics:")
                print(df['final_weighted_sentiment'].describe())
                
                # Visualize results
                #visualize_sentiment_analysis(df, ticker_id)
                
                # Additional analysis: Most influential tweets
                if 'final_weighted_sentiment' in df.columns:
                    print("\nTop 5 Most Positive Influential Tweets:")
                    top_positive = df.sort_values('final_weighted_sentiment', ascending=False).head(5)
                    for idx, row in top_positive.iterrows():
                        print(f"Score: {row['final_weighted_sentiment']:.4f} | {row['tweet_text']}")
                    
                    print("\nTop 5 Most Negative Influential Tweets:")
                    top_negative = df.sort_values('final_weighted_sentiment').head(5)
                    for idx, row in top_negative.iterrows():
                        print(f"Score: {row['final_weighted_sentiment']:.4f} | {row['tweet_text']}")
                        
                return df
            else:
                print("Error: Failed to load Twitter-RoBERTa model")
                return df
        else:
            print("Error: No tweets loaded from database")
            return df

    def get_tweets_sentiment_analysis(self,ticker_id=None, ticker_symbol=None, specific_date=None, start_date=None, end_date=None):

        # Handle ticker symbol conversion to ticker_id if needed
        if ticker_id is None and ticker_symbol:
            # Convert symbol to ID using reverse lookup
            ticker_id = next((id for id, symbol in self.ticker_mapping.items() if symbol == ticker_symbol), 0)
        
        # Handle date parameters
        if specific_date:
            start_date = specific_date
            end_date = specific_date
        
        # Analyze or load the data
        df = self.analyze_tweets_for_date_range(
            specific_date=specific_date,
            start_date=start_date,
            end_date=end_date,
            ticker_id=ticker_id
        )
        
        # If dataframe is empty or less than min_tweet_count, return zeros/defaults
        if df.empty or len(df) < self.min_tweet_count:
            return {
                'tweets_sentiment_score': 0.0,
                'tweets_count': 0,
                'most_positive_tweet': "",
                'most_negative_tweet': "",
                'tweets_weighted_sentiment_score': 0.0,
            }
        
        # Calculate the metrics
        # 1. Average raw sentiment score
        avg_sentiment = df['sentiment_score'].mean()
        
        # 2. Tweet count
        tweet_count = len(df)
        
        # 3. Find most positive tweet
        most_positive_idx = df['final_weighted_sentiment'].idxmax()
        most_positive_tweet = df.loc[most_positive_idx, 'tweet_text']
        
        # 4. Find most negative tweet
        most_negative_idx = df['final_weighted_sentiment'].idxmin()
        most_negative_tweet = df.loc[most_negative_idx, 'tweet_text']
        
        # 5. Average weighted sentiment
        avg_weighted_sentiment = df['final_weighted_sentiment'].mean()
        
        self.db.close()  # Close the database connection
        # Return the metrics
        return {
            'tweets_sentiment_score': float(avg_sentiment),
            'tweets_count': int(tweet_count),
            'most_positive_tweet': str(most_positive_tweet),
            'most_negative_tweet': str(most_negative_tweet),
            'tweets_weighted_sentiment_score': float(avg_weighted_sentiment),
        }
# Example usage
if __name__ == "__main__":
    # Example parameters
    specific_date = '2025-04-10 00:00:00'  # Use specific_date for a single day
    # Alternative: use date range
    # start_date = '2025-01-01' 
    # end_date = '2025-01-31'
    twitter_api = TwitterAPI()
    ticker_id = 3  # AAPL
    summary = twitter_api.get_tweets_sentiment_analysis(ticker_id=ticker_id, specific_date=specific_date)
    print("\n\n============ Sentiment Summary ===========")
    print(summary)
    # Analyze tweets
    # df = analyze_tweets_for_date_range(specific_date=specific_date, ticker_id=ticker_id)
    # Alternative: df = analyze_tweets_for_date_range(start_date=start_date, end_date=end_date, ticker_id=ticker_id)
    