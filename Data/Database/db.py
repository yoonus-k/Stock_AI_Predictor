# Database Handler Module
# 
# This module is used to connect to the database, create tables, and store/retrieve data.
# It provides functionality for managing stock data, patterns, clusters, news articles, 
# tweets, and user authentication.

#################################################################################
# IMPORTS
#################################################################################
import sys
import os
from pathlib import Path

# Third-party imports
import bcrypt
import json
import numpy as np
import pandas as pd
import sqlite3 as db
import sqlitecloud
from datetime import datetime, timedelta



# Setup path
# Get the current working directory (where the notebook/script is running)
current_dir = Path(os.getcwd())
# Navigate to the 'main' folder (adjust if needed)
main_dir = str(current_dir.parent)  # If notebook is inside 'main'
# OR if notebook is outside 'main':
# main_dir = str(current_dir / "main")  # Assumes 'main' is a subfolder
sys.path.append(main_dir)

#################################################################################
# CONSTANTS
#################################################################################
companies = {
    1: "GOLD (XAUUSD)",
    2: "BTC (BTCUSD)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

#################################################################################
# DATABASE CLASS
#################################################################################
class Database:
    """
    Database class to handle all database operations including connection,
    data storage, retrieval, and analysis.
    """
    def __init__(self, db_name='Data/Storage/data.db'):
        """
        Initialize the database connection.
        
        Args:
            db_name (str): Path to the SQLite database file
        """
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.connect()
     
    def connect(self):
        """Connect to the SQLite database."""
        self.connection = db.connect(self.db_name, check_same_thread=False)
        self.cursor = self.connection.cursor()
        print(f"Connected to offline sqlite database")

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print(f"Closed connection to database")
    
    #############################################################################
    # AUTHENTICATION FUNCTIONS
    #############################################################################        
    def login(self, username, password):
        """
        Authenticate a user with the provided credentials.
        
        Args:
            username (str): The username to authenticate
            password (str): The password to verify
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        # Check if the user exists in the database
        user = self.connection.execute('''
            SELECT Password FROM users WHERE Username = ?
        ''', (username,)).fetchone()
        
        if user:
            # Check if the password is correct
            if bcrypt.checkpw(password.encode('utf-8'), user[0]):
                return True
            else:
                return False
        else:
            return False
    
    def get_user_email(self, username):
        """
        Get the email address for a given username.
        
        Args:
            username (str): The username to look up
            
        Returns:
            str: The user's email address
        """
        user_email = self.connection.execute(f'''
            SELECT Email FROM users WHERE Username = '{username}'
        ''').fetchall()
        
        return user_email[0][0]
    
    def get_user_id(self, username):
        """
        Get the user ID for a given username.
        
        Args:
            username (str): The username to look up
            
        Returns:
            int: The user's ID
        """
        user_id = self.connection.execute(f'''
            SELECT UserID FROM users WHERE Username = '{username}'
        ''').fetchall()
        
        return user_id[0][0]
    
    #############################################################################
    # SENTIMENT DATA FUNCTIONS
    #############################################################################    
    def store_articles_in_database(self, articles, most_relevant_stock_id, date):
        """
        Store articles and their stock relations in the database.
        
        Args:
            articles (list): List of article dictionaries to store
            article_stock_relations (list): List of (article_index, relation_dict) tuples
            most_relevant_stock_id (int): ID of the most relevant stock for these articles
            date_str (str): Date string in 'YYYY-MM-DD' format
        """
        try:
            
            cursor = self.connection.cursor()
            
            # Insert articles and get their IDs
            article_ids = []
            
            for article in articles:
                # Create a copy of the article data to modify
                article_data = article.copy()
                
                # Convert authors from list to string if necessary
                if isinstance(article_data.get('authors'), list):
                    article_data['authors'] = ', '.join(article_data['authors'])
                
                   
                # Check if the article already exists in the database
                cursor.execute(
                    "SELECT article_id FROM articles WHERE url = ?", 
                    (article_data['url'],)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Article already exists, just get its ID
                    article_ids.append(existing[0])
                else:
                    # Insert the new article
                    columns = ', '.join(article_data.keys())
                    placeholders = ', '.join(['?' for _ in article_data])
                    
                    insert_query = f"INSERT INTO articles ({columns}) VALUES ({placeholders})"
                    cursor.execute(insert_query, list(article_data.values()))
                    
                    article_ids.append(cursor.lastrowid)
            
            
            # Create or update an entry in the stock_sentiment table
            cursor.execute(
                """
                SELECT COUNT(*) FROM stock_sentiment 
                WHERE stock_id = ? AND date = ?
                """, 
                (most_relevant_stock_id, date)
            )
            
            exists = cursor.fetchone()[0] > 0
          
        
            # get the article count
            article_count = len(article_ids)
            # Calculate aggregate sentiment
            avg_sentiment = np.mean([article['overall_sentiment_score'] for article in articles])
            # Execute the query using today's date for stock_sentiment
            if exists:
                # Update existing entry
                sentiment_update_query = """
                UPDATE stock_sentiment 
                SET article_count = ?, news_sentiment_score = ?
                WHERE stock_id = ? AND date = ?
                """
                cursor.execute(sentiment_update_query, (article_count, avg_sentiment, most_relevant_stock_id, date))
            else:
                # Create new entry
                sentiment_update_query = """
                INSERT INTO stock_sentiment 
                (stock_id, date, article_count, news_sentiment_score)
                VALUES (?, ?, ?, ?)
                """
                cursor.execute(sentiment_update_query, (most_relevant_stock_id, date, article_count, avg_sentiment))
            
            # Commit all changes
            self.connection.commit()
                
        except Exception as e:
            print(f"Database operation error: {e}")
            raise
    
    def get_article_sentiment(self, stock_id, start_date=None, end_date=None, limit=100):
        """
        Get article sentiment data for a specific stock within a date range.
        
        Args:
            stock_id (int): The stock ID to retrieve sentiment for
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            limit (int, optional): Maximum number of articles to return
            
        Returns:
            list: List of dictionaries containing article sentiment data
        """
        query = '''
            SELECT 
                article_id, date, title, summary, source_name, url,
                overall_sentiment_score, overall_sentiment_label,
                most_relevant_stock_sentiment_score, most_relevant_stock_sentiment_label,
                stock_relations
            FROM 
                articles
            WHERE 
                most_relevant_stock_id = ?
        '''
        
        params = [stock_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        
        results = []
        
        for row in self.connection.execute(query, params):
            article = {
                'article_id': row[0],
                'date': row[1],
                'title': row[2],
                'summary': row[3],
                'source': row[4],
                'url': row[5],
                'overall_sentiment_score': row[6],
                'overall_sentiment_label': row[7],
                'stock_sentiment_score': row[8],
                'stock_sentiment_label': row[9]
            }
            
            # Parse stock_relations JSON if available
            try:
                if row[10]:  # stock_relations column
                    article['stock_relations'] = json.loads(row[10])
            except (json.JSONDecodeError, TypeError):
                article['stock_relations'] = []
            
            results.append(article)
            
        return results
    
    def get_stock_sentiment_by_date(self, stock_id, start_date=None, end_date=None):
        """
        Get aggregated stock sentiment data by date.
        
        Args:
            stock_id (int): The stock ID to retrieve sentiment for
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            dict: Dictionary with dates as keys and sentiment data as values
        """
        query = '''
            SELECT 
                date, news_sentiment_score, twitter_sentiment_score, 
                combined_sentiment_score, sentiment_label, 
                article_count, tweet_count
            FROM 
                stock_sentiment
            WHERE 
                stock_id = ?
        '''
        
        params = [stock_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        results = {}
        
        for row in self.connection.execute(query, params):
            date_str = row[0]
            results[date_str] = {
                'news_sentiment': row[1],
                'twitter_sentiment': row[2],
                'combined_sentiment': row[3],
                'sentiment_label': row[4],
                'article_count': row[5],
                'tweet_count': row[6]
            }
            
        return results
    
    #############################################################################
    # STATISTICS FUNCTIONS
    #############################################################################
    def get_statistics(self):
        """
        Get statistics of the clusters and patterns.
        
        Returns:
            dict: Statistics including total clusters, patterns, win rates, etc.
        """
        # Get the clusters and patterns from the database
        clusters = self.get_clusters_all()
        patterns = self.get_patterns_all()
        
        # Calculate statistics
        total_clusters = len(clusters)
        total_patterns = len(patterns)
        avg_patterns_per_cluster = total_patterns / total_clusters if total_clusters > 0 else 0
        avg_win_rate = clusters['ProbabilityScore'].mean() * 100  # Convert to percentage
        avg_max_gain = clusters['MaxGain'].mean() * 100
        avg_max_drawdown = clusters['MaxDrawdown'].mean() * 100
        avg_reward_risk_ratio = clusters['RewardRiskRatio'].mean() 
        avg_profit_factor = clusters['ProfitFactor'].mean()
        
        # Get the best performing cluster   
        best_cluster_idx = clusters['MaxGain'].idxmax()
        best_cluster_return = clusters.loc[best_cluster_idx, 'MaxGain'] * 100
        best_cluster_reward_risk_ratio = clusters.loc[best_cluster_idx, 'RewardRiskRatio']
        best_cluster_profit_factor = clusters.loc[best_cluster_idx, 'ProfitFactor']
        
        # Format the results
        results = {
            "Total Clusters": total_clusters,
            "Total Patterns": total_patterns,
            "Avg Patterns/Cluster": round(avg_patterns_per_cluster, 1),
            "Avg Win Rate": avg_win_rate,
            "Avg Max Gain": f"{round(avg_max_gain, 2)}%",
            "Avg Max Drawdown": f"{round(avg_max_drawdown, 2)}%",
            "Avg Reward/Risk Ratio": round(avg_reward_risk_ratio, 2),
            "Avg Profit Factor": round(avg_profit_factor, 2),
            "Best Cluster Return": f"+{round(best_cluster_return, 1)}%",
            "Best Cluster Reward/Risk Ratio": round(best_cluster_reward_risk_ratio, 2),
            "Best Cluster Profit Factor": round(best_cluster_profit_factor, 2),
        }
        
        return results
    
    #############################################################################
    # FILTER FUNCTIONS
    #############################################################################
    def filter_clusters(self, clusters):
        """
        Filter clusters based on probability score and risk-reward ratio.
        
        Args:
            clusters (DataFrame): DataFrame containing cluster data
            
        Returns:
            DataFrame: Filtered clusters
        """
        # Calculate Reward/Risk Ratio and Profit Factor for all rows at once (vectorized)
        clusters['RewardRiskRatio'] = abs(clusters['MaxGain']) / (abs(clusters['MaxDrawdown']) + 1e-10)  # Avoid division by zero
        clusters['ProfitFactor'] = (clusters['ProbabilityScore'] * clusters['RewardRiskRatio']) / (1 - clusters['ProbabilityScore'] + 1e-10)  # Avoid division by zero
        
        # Filter clusters where ProfitFactor >= 1.0
        filtered_clusters = clusters[clusters['ProfitFactor'] >= 1.0].copy()
        
        return filtered_clusters
            
    #############################################################################
    # DATA STORAGE FUNCTIONS
    #############################################################################     
    def store_stock_data(self, stock_data, stock_ID, stock_symbol, time_frame):
        """
        Store stock data in the database.
        
        Args:
            stock_data (DataFrame): Stock data to store
            stock_ID (int): ID of the stock
            stock_symbol (str): Symbol of the stock
            time_frame (int): Time frame in minutes
        """
        # Insert the data into the table
        for i, (index, row) in enumerate(stock_data.iterrows(), start=0):
            time_Stamp = index.strftime('%Y-%m-%d %H:%M:%S')
            self.connection.execute('''
                INSERT INTO stock_data (StockEntryID, StockID, StockSymbol, Timestamp,TimeFrame ,OpenPrice, ClosePrice, HighPrice, LowPrice, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (i, stock_ID, stock_symbol, time_Stamp, time_frame, row['Open'], row['Close'], row['High'], row['Low'], row['Volume'])) 
        
        # Commit the changes
        self.connection.commit()
        print(f"Stored stock data for {stock_ID} TimeFrame: {time_frame} in database.")
        
    def store_pattern_data(self, stock_ID, pip_pattern_miner):
        """
        Store pattern data in the database.
        
        Args:
            stock_ID (int): ID of the stock
            pip_pattern_miner: Pattern miner object containing pattern data
        """
        # Store patterns in the database and table patterns
        for i, pattern in enumerate(pip_pattern_miner._unique_pip_patterns):
            # Convert the pattern to string
            pattern_str = ','.join([str(x) for x in pattern])
            # Get the time span of the pattern
            time_span = pip_pattern_miner._lookback
            
            # Determine market condition (Bullish, Bearish, Neutral)
            first_point = pattern[0]
            last_point = pattern[-1]
            
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
                
            # Get pattern return and determine label
            pattern_ruturn = pip_pattern_miner._returns_fixed_hold[i]
            
            if pattern_ruturn > 0:
                pattern_label = 'Buy'
            elif pattern_ruturn < 0:
                pattern_label = 'Sell'
            else:
                pattern_label = 'Neutral'
                
            # Calculate max gain and drawdown based on pattern label
            if pattern_label == 'Buy':
                pattern_max_gain = pip_pattern_miner._returns_mfe[i]
                pattern_max_drawdown = pip_pattern_miner._returns_mae[i]
            elif pattern_label == 'Sell':
                pattern_max_gain = pip_pattern_miner._returns_mae[i]
                pattern_max_drawdown = pip_pattern_miner._returns_mfe[i]
            else:
                pattern_max_gain = 0
                pattern_max_drawdown = 0
                    
            # Insert the data into the table
            self.connection.execute('''
                INSERT INTO patterns (PatternID, StockID, PricePoints, TimeSpan, MarketCondition, Outcome, Label, MaxGain, MaxDrawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (i, stock_ID, pattern_str, time_span, market_condition, pattern_ruturn, pattern_label, pattern_max_gain, pattern_max_drawdown))
            
        # Commit the changes
        self.connection.commit()
        
    def store_cluster_data(self, stock_ID, pip_pattern_miner):
        """
        Store cluster data in the database.
        
        Args:
            stock_ID (int): ID of the stock
            pip_pattern_miner: Pattern miner object containing cluster data
        """
        # Store clusters in the database
        for i, cluster in enumerate(pip_pattern_miner._cluster_centers):
            # Convert the cluster to string
            cluster_str = ','.join([str(x) for x in cluster])
            
            # Determine market condition (Bullish, Bearish, Neutral)
            first_point = cluster[0]
            last_point = cluster[-1]
            
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
                
            # Get cluster return and determine label
            cluster_ruturn = pip_pattern_miner._cluster_returns[i]
            
            if cluster_ruturn > 0:
                cluster_label = 'Buy'
            elif cluster_ruturn < 0:
                cluster_label = 'Sell'
            else:
                cluster_label = 'Neutral'
                
            # Get the pattern count of the cluster
            pattern_count = len(pip_pattern_miner._pip_clusters[i])
            
            # Calculate max gain and drawdown based on cluster label
            if cluster_label == 'Buy':
                cluster_max_gain = pip_pattern_miner._cluster_mfe[i]
                cluster_max_drawdown = pip_pattern_miner._cluster_mae[i]
            elif cluster_label == 'Sell':
                cluster_max_gain = pip_pattern_miner._cluster_mae[i]
                cluster_max_drawdown = pip_pattern_miner._cluster_mfe[i]
            else:
                cluster_max_gain = 0
                cluster_max_drawdown = 0
                
            # Insert the data into the table
            self.connection.execute('''
                INSERT INTO clusters (ClusterID, StockID, AVGPricePoints, MarketCondition, Outcome, Label, Pattern_Count, MaxGain, MaxDrawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (i, stock_ID, cluster_str, market_condition, cluster_ruturn, cluster_label, pattern_count, cluster_max_gain, cluster_max_drawdown))
           
        # Commit the changes
        self.connection.commit()
    
    def bind_pattern_cluster(self, stock_ID, pip_pattern_miner):
        """
        Bind pattern and cluster data. The patterns table contains a foreign key to the clusters table.
        
        Args:
            stock_ID (int): ID of the stock
            pip_pattern_miner: Pattern miner object containing pattern and cluster data
        """
        # Loop through the clusters and update patterns table
        for i, cluster in enumerate(pip_pattern_miner._pip_clusters):
            # Update the Pattern_Count in the clusters table
            self.connection.execute('''
                UPDATE clusters
                SET Pattern_Count = ?
                WHERE ClusterID = ? AND StockID = ?
                ''', (len(cluster), i, stock_ID))
            
            # Update cluster ID in the patterns table
            for pattern in cluster:
                self.connection.execute('''
                    UPDATE patterns
                    SET ClusterID = ?
                    WHERE PatternID = ? AND StockID = ?
                ''', (i, pattern, stock_ID))
                
        # Commit the changes
        self.connection.commit()
        
    def store_prediction_data(self, stock_ID, data):
        """
        Store prediction data in the database.
        
        Args:
            stock_ID (int): ID of the stock
            data (dict): Prediction data
            
        Returns:
            int: ID of the inserted prediction
        """
        res = self.connection.execute('''
            INSERT INTO Predictions (StockID, PatternID, NewsID, TweetID, PredictionDate, PredictedOutcome, ConfidenceLevel)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (stock_ID, data['pattern_metrics']['pattern_id'], 0, 0, data['date'], json.dumps(data), data['confidence']))
                              
        # Get the prediction id
        prediction_id = res.lastrowid
        # Commit the changes
        self.connection.commit()
        
        return prediction_id
        
    def store_notification_data(self, user_id, prediction_id, sent_time, notification_type, status):
        """
        Store notification data in the database.
        
        Args:
            user_id (int): ID of the user
            prediction_id (int): ID of the prediction
            sent_time (str): Time the notification was sent
            notification_type (str): Type of notification
            status (str): Status of the notification
        """
        # Insert data into the Notifications table
        self.connection.execute('''
            INSERT INTO Notifications (UserID, PredictionID, SentTime, NotificationType, Status)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, prediction_id, sent_time, notification_type, status))
        
        # Commit the changes
        self.connection.commit()

    def store_live_news_articles(self, date, authors, source_domain, source_name, title, summary, url, topics,
                                 ticker_sentiment, overall_sentiment_label, overall_sentiment_score, event_type,
                                 sentiment_label, sentiment_score, fetch_timestamp):
        """
        Store live news articles in the database.
        
        Args:
            Multiple parameters for news article data
        """
        try:
            self.connection.execute('''
                INSERT INTO live_articles (Date, Authors, Source_Domain, Source_Name, Title, Summary, Url, Topics, 
                                         Ticker_Sentiment, Overall_Sentiment_Label, Overall_Sentiment_Score, 
                                         Event_Type, Sentiment_Label, Sentiment_Score, Fetch_Timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, authors, source_domain, source_name, title, summary, url, topics, ticker_sentiment,
                  overall_sentiment_label, overall_sentiment_score, event_type, sentiment_label, sentiment_score,
                  fetch_timestamp))
            self.connection.commit()
        except sqlitecloud.Error as e:
            print(f"An error occurred: {e}")

    def store_tweets(self, ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count,
                     quote_count, bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term, author_username,
                     author_name, author_verified, author_blue_verified, author_followers,
                     author_following, sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment,
                     collected_at):
        """
        Store tweets in the database.
        
        Args:
            Multiple parameters for tweet data
        """
        try:
            self.connection.execute('''
                INSERT INTO tweets (ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count, 
                                  quote_count, bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term, 
                                  author_username, author_name, author_verified, author_blue_verified, author_followers, 
                                  author_following, sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment, 
                                  collected_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count,
                            quote_count, bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term,
                            author_username, author_name, author_verified, author_blue_verified,
                            author_followers, author_following, sentiment_label, sentiment_score,
                            sentiment_magnitude, weighted_sentiment, collected_at))
            self.connection.commit()
        except sqlitecloud.Error as e:
            return
            print(f"An error occurred: {e}")
    
    #############################################################################
    # DATA RETRIEVAL FUNCTIONS
    #############################################################################
    def get_stock_data(self, stock_ID, time_frame=60):
        """
        Get stock data from the database.
        
        Args:
            stock_ID (int): ID of the stock
            time_frame (int): Time frame in minutes
            
        Returns:
            DataFrame: Stock data
        """
        # Get stock data from the database
        stock_data = self.connection.execute(f'''
            SELECT * FROM stock_data WHERE StockID = {stock_ID} AND TimeFrame = {time_frame}
        ''').fetchall()
        
        # Convert to DataFrame and process
        stock_data = pd.DataFrame(stock_data, columns=['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice'])
        stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
        stock_data.set_index('Timestamp', inplace=True)
        stock_data.sort_index(inplace=True)
        
        return stock_data
    
    def get_stock_data_range(self, stock_ID, start_date, end_date, time_frame=60):
        """
        Get stock data for a specific date range.
        
        Args:
            stock_ID (int): ID of the stock
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            time_frame (int): Time frame in minutes
            
        Returns:
            DataFrame: Stock data within the specified range
        """
        # Get stock data from the database for the specified range
        stock_data = self.connection.execute(f'''
            SELECT * 
            FROM stock_data 
            WHERE 
                StockID = {stock_ID}
                AND Timestamp >= '{start_date}'
                AND Timestamp <= '{end_date}'
                AND TimeFrame = {time_frame}
        ''').fetchall()
        
        # Convert to DataFrame and process
        stock_data = pd.DataFrame(stock_data, columns=['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice'])
        stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
        stock_data.set_index('Timestamp', inplace=True)
        stock_data.sort_index(inplace=True)     
        
        return stock_data

    def get_stock_data_by_date_and_period(self, stock_ID, date, period):
        """
        Get stock data for a specific date and period.
        
        Args:
            stock_ID (int): ID of the stock
            date (str): Date in format 'YYYY-MM-DD'
            period (int): Number of days back to retrieve data
            
        Returns:
            DataFrame: Stock data for the specified period
        """
        from_date = pd.to_datetime(date) - pd.Timedelta(days=period)
        to_date = pd.to_datetime(date)
        
        # Get stock data from the database
        stock_data = self.connection.execute(f'''
            SELECT * FROM stock_data WHERE StockID = {stock_ID} AND Timestamp BETWEEN '{from_date}' AND '{to_date}'
        ''').fetchall()
        
        # Convert to DataFrame and process
        stock_data = pd.DataFrame(stock_data, columns=['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice'])
        stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
        stock_data.set_index('Timestamp', inplace=True)
        stock_data.sort_index(inplace=True)
        
        return stock_data
    
    def get_patterns_by_stock_id(self, stock_ID):
        """
        Get patterns for a specific stock.
        
        Args:
            stock_ID (int): ID of the stock
            
        Returns:
            DataFrame: Patterns for the specified stock
        """
        # Get patterns from the database
        patterns = self.connection.execute(f'''
            SELECT * FROM patterns WHERE StockID = {stock_ID}
        ''').fetchall()
        
        # Convert to DataFrame
        patterns = pd.DataFrame(patterns, columns=['PatternID', 'StockID', 'ClusterID', 'PricePoints', 'TimeSpan', 'MarketCondition', 'Outcome', 'Label', 'MaxGain', 'MaxDrawdown'])
 
        return patterns
    
    def get_patterns_all(self):
        """
        Get all patterns from the database.
        
        Returns:
            DataFrame: All patterns
        """
        # Get all patterns from the database
        patterns = self.connection.execute(f'''
            SELECT * FROM patterns
        ''').fetchall()
        
        # Convert to DataFrame
        patterns = pd.DataFrame(patterns, columns=['PatternID', 'StockID', 'ClusterID', 'PricePoints', 'TimeSpan', 'MarketCondition', 'Outcome', 'Label', 'MaxGain', 'MaxDrawdown'])
 
        return patterns
    
    def get_clusters_by_stock_id(self, stock_ID):
        """
        Get clusters for a specific stock.
        
        Args:
            stock_ID (int): ID of the stock
            
        Returns:
            DataFrame: Filtered clusters for the specified stock
        """
        # Get clusters from the database
        clusters = self.connection.execute(f'''
            SELECT * FROM clusters WHERE StockID = {stock_ID}
        ''').fetchall()
        
        # Convert to DataFrame
        clusters = pd.DataFrame(clusters, columns=['ClusterID', 'StockID', 'AVGPricePoints', 'MarketCondition', 'Outcome', 'Label','ProbabilityScore',  'Pattern_Count', 'MaxGain', 'MaxDrawdown'])
       
        return self.filter_clusters(clusters)
    
    def get_clusters_all(self):
        """
        Get all clusters from the database.
        
        Returns:
            DataFrame: Filtered clusters
        """
        # Get all clusters from the database
        clusters = self.connection.execute(f'''
            SELECT * FROM clusters 
        ''').fetchall()
        
        # Convert to DataFrame
        clusters = pd.DataFrame(clusters, columns=['ClusterID', 'StockID', 'AVGPricePoints', 'MarketCondition', 'Outcome', 'Label','ProbabilityScore',  'Pattern_Count', 'MaxGain', 'MaxDrawdown'])
       
        return self.filter_clusters(clusters)

    def get_cluster_probability_score(self, stock_id, cluster_id):
        """
        Calculate the probability score for a cluster.
        
        Args:
            stock_id (int): ID of the stock
            cluster_id (int): ID of the cluster
            
        Returns:
            float: Probability score
        """
        # Get patterns belonging to the cluster
        patterns = self.connection.execute(f'''
            SELECT Outcome FROM patterns WHERE ClusterID = {cluster_id} AND StockID = {stock_id}
        ''').fetchall()
        
        # Convert to DataFrame
        patterns = pd.DataFrame(patterns, columns=['Outcome'])
        
        # Calculate statistics
        total_positive_returns = (patterns['Outcome'] > 0).sum()
        total_negative_returns = (patterns['Outcome'] < 0).sum()
        total_patterns = len(patterns)
       
        # Get cluster label
        cluster_label = pd.read_sql_query(f'''
            SELECT Label FROM clusters WHERE ClusterID = {cluster_id} AND StockID = {stock_id}
        ''', self.connection)
        
        # Calculate probability score based on label
        if cluster_label.iloc[0]['Label'] == 'Buy':
            probability_score = total_positive_returns / total_patterns
        elif cluster_label.iloc[0]['Label'] == 'Sell':
            probability_score = total_negative_returns / total_patterns
        else:
            probability_score = 0.5
            
        return probability_score
    
    def get_prediction_data(self, stock_ID):
        """
        Get prediction data for a specific stock.
        
        Args:
            stock_ID (int): ID of the stock
            
        Returns:
            DataFrame: Prediction data
        """
        # Get prediction data from the database
        prediction_data = self.connection.execute(f'''
            SELECT * FROM Prediction WHERE stock_id = {stock_ID}
        ''').fetchall()
        
        # Convert to DataFrame
        prediction_data = pd.DataFrame(prediction_data, columns=['PredictionID','StockID' ,'PatternID', 'SentimentID','PredictionData' ,'PredictedOutcome' ,'ConfidenceLevel'])
        
        # Parse JSON data
        prediction_data['PredictionData'] = prediction_data['PredictionData'].apply(json.loads)
        
        return prediction_data
    
    def get_notification_data(self, user_id):
        """
        Get notification data for a specific user.
        
        Args:
            user_id (int): ID of the user
            
        Returns:
            DataFrame: Notification data
        """
        # Get notification data from the database
        notification_data = self.connection.execute(f'''
            SELECT * FROM Notifications WHERE UserID = {user_id}
        ''').fetchall()
        
        # Convert to DataFrame
        notification_data = pd.DataFrame(notification_data, columns=['NotificationID', 'UserID', 'PredictionID', 'SentTime', 'NotificationType', 'Status'])
        
        return notification_data

    def get_news_articles(self):
        """
        Get all news articles from the database.
        
        Returns:
            DataFrame: News articles
        """
        # Get news articles from the database
        news_articles = self.connection.execute('''
            SELECT * FROM articles
        ''').fetchall()
        
        # Convert date to datetime
        news_articles['Date'] = pd.to_datetime(news_articles['Date'])
        
        return news_articles

    def get_live_news_articles(self):
        """
        Get live news articles from the database.
        
        Returns:
            DataFrame: Live news articles
        """
        # Get live news articles from the database
        live_news_articles = self.connection.execute('''
            SELECT * FROM live_articles WHERE ID > 12153
        ''').fetchall()
        
        # Convert to DataFrame
        live_news_articles = pd.DataFrame(live_news_articles, columns=['ID', 'Date', 'Authors', 'Source_Domain', 'Source_Name', 'Title', 'Summary', 'Url', 'Topics', 'Ticker_Sentiment', 'Overall_Sentiment_Label', 'Overall_Sentiment_Score', 'Event_Type', 'Sentiment_Label', 'Sentiment_Score', 'Fetch_Timestamp'])
        
        # Convert date to datetime
        live_news_articles['Date'] = pd.to_datetime(live_news_articles['Date'])
        
        return live_news_articles

    def get_tweets(self):
        """
        Get all tweets from the database.
        
        Returns:
            DataFrame: Tweets
        """
        # Get tweets from the database
        tweets = self.connection.execute('''
            SELECT * FROM tweets
        ''').fetchall()
        
        # Convert to DataFrame
        tweets = pd.DataFrame(tweets, columns=['ID', 'ticker_id', 'tweet_id', 'tweet_text', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count', 'lang', 'is_reply', 'is_quote', 'is_retweet', 'url', 'search_term', 'author_username', 'author_name', 'author_verified', 'author_blue_verified', 'author_followers', 'author_following', 'sentiment_label', 'sentiment_score', 'sentiment_magnitude', 'weighted_sentiment', 'collected_at'])
        
        # Convert date to datetime
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
        
        return tweets

    def get_tweets_by_date_and_ticker(self, start_date, end_date, ticker_id):
        """
        Get tweets for a specific date range and ticker.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            ticker_id (int): ID of the ticker
            
        Returns:
            DataFrame: Tweets for the specified date range and ticker
        """
        # Get tweets from the database
        tweets = self.connection.execute('''
            SELECT * FROM tweets
            WHERE created_at BETWEEN ? AND ? AND ticker_id = ?
        ''', (start_date, end_date, ticker_id)).fetchall()
        
        # Convert to DataFrame
        tweets = pd.DataFrame(tweets, columns=['ID', 'ticker_id', 'tweet_id', 'tweet_text', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count', 'lang', 'is_reply', 'is_quote', 'is_retweet', 'url', 'search_term', 'author_username', 'author_name', 'author_verified', 'author_blue_verified', 'author_followers', 'author_following', 'sentiment_label', 'sentiment_score', 'sentiment_magnitude', 'weighted_sentiment', 'collected_at'])
        
        return tweets

    def get_tweets_count_by_date_and_ticker(self, start_date, end_date, ticker_id):
        """
        Get tweet counts by date and ticker.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            ticker_id (int): ID of the ticker
            
        Returns:
            DataFrame: Tweet counts grouped by date
        """
        # Get tweet counts grouped by date
        tweets = self.connection.execute('''
            SELECT DATE(created_at) AS date, ticker_id, COUNT(*) AS tweet_count
            FROM tweets
            WHERE created_at BETWEEN ? AND ? AND ticker_id = ?
            GROUP BY DATE(created_at), ticker_id
            ORDER BY DATE(created_at) ASC
        ''', (start_date, end_date, ticker_id)).fetchall()
        
        # Convert to DataFrame
        tweets = pd.DataFrame(tweets, columns=['date', 'ticker_id', 'tweet_count'])
        
        return tweets
    
    #############################################################################
    # UPDATE FUNCTIONS
    #############################################################################
    def update_cluster_probability_score_based_on_patterns(self, stock_id, cluster_id):
        """
        Update the probability score for a cluster based on its patterns.
        
        Args:
            stock_id (int): ID of the stock
            cluster_id (int): ID of the cluster
            
        Returns:
            float: Updated probability score
        """
        probability_score = self.get_cluster_probability_score(stock_id, cluster_id)
        
        # Update the cluster probability score
        self.connection.execute('''
            UPDATE clusters
            SET ProbabilityScore = ?
            WHERE ClusterID = ? AND StockID = ?
        ''', (probability_score, cluster_id, stock_id))
        
        # Commit the changes
        self.connection.commit()
        
        return probability_score
    
    def update_all_cluster_probability_score(self, stock_id, pip_pattern_miner):
        """
        Update all cluster probability scores for a stock.
        
        Args:
            stock_id (int): ID of the stock
            pip_pattern_miner: Pattern miner object
        """
        # Update probability scores for all clusters
        for i in range(len(pip_pattern_miner._cluster_centers)):
            self.update_cluster_probability_score_based_on_patterns(stock_id, i)

    def update_tweet_sentiment(self, tweet_id, sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment):
        """
        Update sentiment data for an existing tweet.
        
        Args:
            tweet_id (str): ID of the tweet
            sentiment_label (str): Sentiment label (positive, negative, neutral)
            sentiment_score (float): Sentiment score
            sentiment_magnitude (float): Sentiment magnitude
            weighted_sentiment (float): Weighted sentiment score
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.connection.execute('''
                UPDATE tweets 
                SET sentiment_label = ?, 
                    sentiment_score = ?, 
                    sentiment_magnitude = ?, 
                    weighted_sentiment = ?
                WHERE tweet_id = ?
            ''', (sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment, tweet_id))
            
            # Check if any rows were affected
            rows_affected = self.connection.total_changes
            if rows_affected == 0:
                print(f"Warning: No tweet found with ID {tweet_id}")
                return False
                
            self.connection.commit()
            return True
            
        except sqlitecloud.Error as e:
            print(f"Database error updating tweet {tweet_id}: {e}")
            return False

    #############################################################################
    # STOCK MANAGEMENT FUNCTIONS
    #############################################################################
    def get_stock_id(self, symbol):
        """
        Get the stock ID for a given symbol.
        
        Args:
            symbol (str): The stock symbol to look up
            
        Returns:
            int: The stock ID, or None if not found
        """
        return self.stock_manager.get_stock_id(symbol)
    
    def get_stock_symbol(self, stock_id):
        """
        Get the stock symbol for a given ID.
        
        Args:
            stock_id (int): The stock ID to look up
            
        Returns:
            str: The stock symbol, or None if not found
        """
        return self.stock_manager.get_stock_symbol(stock_id)
    
    def get_stocks(self):
        """
        Get all stocks as a dictionary of ID to symbol mappings.
        
        Returns:
            dict: Dictionary with stock IDs as keys and symbols as values
        """
        return self.stock_manager.get_all_stocks()
    
    def get_api_symbol(self, symbol):
        """
        Get the API-specific symbol for a given internal symbol.
        
        Args:
            symbol (str): The internal stock symbol
            
        Returns:
            str: The API-specific symbol, or the original symbol if no mapping exists
        """
        return self.stock_manager.get_api_symbol(symbol)
    
    def get_internal_symbol(self, api_symbol):
        """
        Get the internal symbol for a given API-specific symbol.
        
        Args:
            api_symbol (str): The API-specific symbol
            
        Returns:
            str: The internal symbol, or the original API symbol if no mapping exists
        """
        return self.stock_manager.get_internal_symbol(api_symbol)
    
    def add_stock(self, symbol, stock_id=None):
        """
        Add a new stock to the system.
        
        Args:
            symbol (str): The stock symbol to add
            stock_id (int, optional): The stock ID to use. If None, uses the next available ID.
            
        Returns:
            int: The stock ID that was assigned
        """
        return self.stock_manager.add_stock(symbol, stock_id)

#############################################################################
# MAIN EXECUTION
#############################################################################
# Main function to create the database and tables
if __name__ == '__main__':
    db = Database()
    data = db.get_stock_data_by_date_and_period(3, '2023-01-01', 5)
    
    date_datetime = pd.to_datetime('2023-01-01')
    start_time = date_datetime - pd.Timedelta(hours=25)
    window = data.loc[(data.index >= start_time) & (data.index <= date_datetime)]
    if window.empty:
            start_time = date_datetime - pd.Timedelta(hours=50)
            window = data.loc[(data.index >= start_time) & (data.index <= date_datetime)]
    window_prices = np.array(window['ClosePrice'])
    print(window_prices.flatten())
    db.close()



