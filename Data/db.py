# this model is used to connect to the database and create the tables also stores the data
import sys
import os
from pathlib import Path

import bcrypt
import json
# Get the current working directory (where the notebook/script is running)
current_dir = Path(os.getcwd())
# Navigate to the 'main' folder (adjust if needed)
main_dir = str(current_dir.parent)  # If notebook is inside 'main'
# OR if notebook is outside 'main':
# main_dir = str(current_dir / "main")  # Assumes 'main' is a subfolder
sys.path.append(main_dir)
import sqlite3 as db
import pandas as pd

companies = {
    1: "GOLD (XAUUSD)",
    2: "BTC (BTCUSD)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}


class Database:
    def __init__(self, db_name='Data/data.db'):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.connect()
     

    def connect(self):
        """Connect to the SQLite database."""
        self.connection = db.connect(self.db_name)
        self.cursor = self.connection.cursor()
        print(f"Connected to offline sqlite database")

   
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print(f"Closed connection to database")
            
         
            
    # function to simulate the user login to the database
    def login(self, username, password):
        # check if the user exists in the database
        """Authenticate a user"""
        user = self.connection.execute('''
            SELECT Password FROM users WHERE Username = ?
        ''', (username,)).fetchone()
        
        if user:
           # print(f"Hashed password: {user[0]}")
            # check if the password is correct
            if bcrypt.checkpw(password.encode('utf-8'), user[0]):
                #print(f"User {username} logged in successfully.")
                return True
            else:
                #print(f"Invalid username or password.")
                return False
        else:
            #print(f"Invalid username or password.")
            return False
        
    ##### -------- STAT Functions -------- #####
    # funtion to get the statistics of the clusters and patterns
    def get_statistics(self):
        # get the statistics of the clusters and patterns (Total Clusters, Total Patterns, Avg Patterns/Cluster, Avg Win Rate, Avg Max Gain, Avg Max Drawdown, Avg Reward/Risk, Best Cluster Return)
        # cluster label distribution (Buy, Sell, Neutral)
        # best performing cluster and the worst performing cluster
        
        # get the clusters and patterns from the database
        clusters = self.get_clusters_all()
        patterns = self.get_patterns_all()
        # get the total number of clusters and patterns
        total_clusters = len(clusters)
        total_patterns = len(patterns)
        # get the average number of patterns per cluster
        avg_patterns_per_cluster = total_patterns / total_clusters if total_clusters > 0 else 0
        # get the average win rate
        avg_win_rate = clusters['ProbabilityScore'].mean() * 100  # Convert to percentage
        # get the average max gain
        avg_max_gain = clusters['MaxGain'].mean() * 100
        # get the average max drawdown
        avg_max_drawdown = clusters['MaxDrawdown'].mean() * 100
        # get the average reward/risk ratio
        avg_reward_risk_ratio = clusters['RewardRiskRatio'].mean() 
        # get the average profit factor
        avg_profit_factor = clusters['ProfitFactor'].mean()
        
        # get the best performing cluster   
        best_cluster_idx = clusters['MaxGain'].idxmax()
        best_cluster_return = clusters.loc[best_cluster_idx, 'MaxGain'] * 100
        best_cluster_reward_risk_ratio = clusters.loc[best_cluster_idx, 'RewardRiskRatio']
        best_cluster_profit_factor = clusters.loc[best_cluster_idx, 'ProfitFactor']
        
        # format the results
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
    
    
    ##### -------- Filter Functions -------- #####
    ##### -------------------------------- #####
    # function to filter the clusters based on the probability score and the risk reward ratio
    def filter_clusters(self, clusters):
        # Calculate Reward/Risk Ratio and Profit Factor for all rows at once (vectorized)
        clusters['RewardRiskRatio'] = abs(clusters['MaxGain']) / (abs(clusters['MaxDrawdown']) + 1e-10)  # Avoid division by zero
        clusters['ProfitFactor'] = (clusters['ProbabilityScore'] * clusters['RewardRiskRatio']) / (1 - clusters['ProbabilityScore'] + 1e-10)  # Avoid division by zero
        
        # Filter clusters where ProfitFactor >= 1.0
        filtered_clusters = clusters[clusters['ProfitFactor'] >= 1.0].copy()
        
        return filtered_clusters
            
    ##### -------- Store Functions -------- #####      
    ##### -------------------------------- #####     
     # funtion to store the stock data in the database
    def store_stock_data(self, stock_data, stock_ID, stock_symbol,time_frame):
        
        # insert the data into the table
        for i, (index, row) in enumerate(stock_data.iterrows(), start=0):
            time_Stamp = index.strftime('%Y-%m-%d %H:%M:%S')
            self.connection.execute('''
                INSERT INTO stock_data (StockEntryID, StockID, StockSymbol, Timestamp,TimeFrame ,OpenPrice, ClosePrice, HighPrice, LowPrice, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (i, stock_ID, stock_symbol, time_Stamp, time_frame, row['Open'], row['Close'], row['High'], row['Low'], row['Volume'])) 
        # commit the changes
        self.connection.commit()
        print(f"Stored stock data for {stock_ID} TimeFrame: {time_frame} in database.")
        
     # funtion to store pattern data in the database
    def store_pattern_data(self,  stock_ID, pip_pattern_miner ):
        # store these pattern in the database data.db and table patterns
        # insert the data into the table
        for i, pattern in enumerate(pip_pattern_miner._unique_pip_patterns):
            # convert the cluster to string
            pattern_str = ','.join([str(x) for x in pattern])
             # get the time span of the cluster
            time_span = pip_pattern_miner._lookback
            # get the market condition of the patter, from the first point of the pattern to the last point
            # get the first point of the pattern
            first_point = pattern[0]
            # get the last point of the pattern
            last_point = pattern[-1]
            # get the market condition of the pattern
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
                
            # get the return of the pattern
            pattern_ruturn = pip_pattern_miner._returns_fixed_hold[i]
            # get the pattern Label, if return is greater than 0 then it is a buy pattern else it is a sell pattern
            if pattern_ruturn > 0:
                pattern_label = 'Buy'
            elif pattern_ruturn < 0:
                pattern_label = 'Sell'
            else:
                pattern_label = 'Neutral'
                
            # if it's a buy , then the MaxGain will be the mfe and the MaxDrawdown will be the mae
            # if it's a sell , then the MaxGain will be the mae and the MaxDrawdown will be the mfe
            if pattern_label == 'Buy':
                pattern_max_gain = pip_pattern_miner._returns_mfe[i]
                pattern_max_drawdown = pip_pattern_miner._returns_mae[i]
            elif pattern_label == 'Sell':
                pattern_max_gain = pip_pattern_miner._returns_mae[i]
                pattern_max_drawdown = pip_pattern_miner._returns_mfe[i]
            else:
                pattern_max_gain = 0
                pattern_max_drawdown = 0
                    
            # insert the data into the table
            self.connection.execute('''
                INSERT INTO patterns (PatternID, StockID, PricePoints , TimeSpan , MarketCondition , Outcome , Label , MaxGain , MaxDrawdown)
                VALUES (?, ?, ? , ?, ?, ?, ?, ?, ?)
            ''', (i, stock_ID,pattern_str , time_span, market_condition , pattern_ruturn , pattern_label, pattern_max_gain , pattern_max_drawdown))
        # commit the changes
        self.connection.commit()
        
        
     # funtion to store the cluster data in the database
    def store_cluster_data(self, stock_ID,pip_pattern_miner):
        # store these pattern in the database data.db and table patterns
        # insert the data into the table
        for i, cluster in enumerate(pip_pattern_miner._cluster_centers):
            # convert the cluster to string
            cluster_str = ','.join([str(x) for x in cluster])
            # get the market condition of the cluster, from the first point of the pattern to the last point
            # get the first point of the cluster
            first_point = cluster[0]
            # get the last point of the cluster
            last_point = cluster[-1]
            # get the market condition of the cluster
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
            # get the return of the cluster
            cluster_ruturn = pip_pattern_miner._cluster_returns[i]
            # get the cluster Label, if return is greater than 0 then it is a buy pattern else it is a sell pattern
            if cluster_ruturn > 0:
                cluster_label = 'Buy'
            elif cluster_ruturn < 0:
                cluster_label = 'Sell'
            else:
                cluster_label = 'Neutral'
                
            # get the pattern count of the cluster
            pattern_count = len(pip_pattern_miner._pip_clusters[i])
            
            # if it's a buy , then the MaxGain will be the mfe and the MaxDrawdown will be the mae
            # if it's a sell , then the MaxGain will be the mae and the MaxDrawdown will be the mfe
            if cluster_label == 'Buy':
                cluster_max_gain = pip_pattern_miner._cluster_mfe[i]
                cluster_max_drawdown = pip_pattern_miner._cluster_mae[i]
            elif cluster_label == 'Sell':
                cluster_max_gain = pip_pattern_miner._cluster_mae[i]
                cluster_max_drawdown = pip_pattern_miner._cluster_mfe[i]
            else:
                cluster_max_gain = 0
                cluster_max_drawdown = 0
                
            # insert the data into the table
            self.connection.execute('''
                INSERT INTO clusters (ClusterID, StockID, AVGPricePoints , MarketCondition , Outcome, Label , Pattern_Count , MaxGain , MaxDrawdown)
                VALUES (?, ?, ? , ?, ?, ?, ?, ? , ?)
            ''', (i,stock_ID, cluster_str , market_condition , cluster_ruturn , cluster_label , pattern_count , cluster_max_gain , cluster_max_drawdown))
           
        # commit the changes
        self.connection.commit()
       
    # funtion to store the prediction data in the database
    def store_prediction_data(self, stock_ID, pip_pattern_miner):
        
        prediction_data = {
            'date': '2023-05-15',
            'stock_id': 'AAPL',
            'stock_name': 'Apple Inc.',
            'current_price': 172.57,
            'pattern_prediction': 185.20,
            'final_prediction': 183.50,
            'confidence': 0.95,
            'action': 'BUY',
            'position_size': 0.15,
            'pattern_metrics': {
                'pattern_id': 42,
                'type': 'Bullish Flag',
                'probability': 0.87,
                'max_gain': 0.12,
                'max_drawdown': 0.05,
                'reward_risk_ratio': 2.4
            },
            'sentiment_metrics': {
                'score': 0.82,
                'magnitude': 0.75
            },
            'hybrid_score': 0.89
        }

        self.connection.execute('''INSERT INTO Prediction (PatternID, SentimentID, PredictedOutcome)
        VALUES (?, ?, ?)''', (123, 456, json.dumps(prediction_data)))
                              
        self.connection.commit()
        
    # funtion to store the notification data in the database
    def store_notification_data(self, user_id, prediction_id, sent_time, notification_type, status):
        # columns are : NotificationID UserID PredictionID SentTime NotificationType Status
        # insert the data into the table
        self.connection.execute('''
            INSERT INTO Notifications (UserID, PredictionID, SentTime, NotificationType, Status)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, prediction_id, sent_time, notification_type, status))
        # commit the changes
        self.connection.commit()

    # function to store the news articles in the database
    # def store_news_articles(self, date, author, source_id, source_name, title, description, url, content,
    #                         event_type,
    #                         sentiment_label, sentiment_score, fetch_timestamp):
    #     try:
    #         cursor = self.cursor
    #         cursor.execute('''
    #             INSERT INTO articles (Date, Author, Source_ID, Source_Name, Title, Description, Url, Content, Event_Type, Sentiment_Label, Sentiment_Score, Fetch_Timestamp)
    #             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #         ''', (
    #         date, author, source_id, source_name, title, description, url, content, event_type, sentiment_label,
    #         sentiment_score,
    #         fetch_timestamp))
    #         self.connection.commit()
    #         print("News articles data stored successfully.")
    #     except sqlitecloud.Error as e:
    #         print(f"An error occurred: {e}")

    # DB (ID, Date, Authors, source_domain, source_name, title, summary, url, topics, ticker_sentiment, overall_sentiment_label, overall_sentiment_score, event_type, sentiment_label, sentiment_score)
    # function to store_live_news_articles
    def store_live_news_articles(self, date, authors, source_domain, source_name, title, summary, url, topics,
                                 ticker_sentiment, overall_sentiment_label, overall_sentiment_score, event_type,
                                 sentiment_label, sentiment_score, fetch_timestamp):
        try:
            self.connection.execute('''
                INSERT INTO live_articles (Date, Authors, Source_Domain, Source_Name, Title, Summary, Url, Topics, Ticker_Sentiment, Overall_Sentiment_Label, Overall_Sentiment_Score, Event_Type, Sentiment_Label, Sentiment_Score, Fetch_Timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, authors, source_domain, source_name, title, summary, url, topics, ticker_sentiment,
                  overall_sentiment_label, overall_sentiment_score, event_type, sentiment_label, sentiment_score,
                  fetch_timestamp))
            self.connection.commit()
            # print("Live news articles data stored successfully.")
        except sqlitecloud.Error as e:
            print(f"An error occurred: {e}")

    # function to store tweets in the database
    def store_tweets(self, ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count,
                     quote_count,
                     bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term, author_username,
                     author_name, author_verified, author_blue_verified, author_followers,
                     author_following, sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment,
                     collected_at):
        try:
            self.connection.execute('''
                INSERT INTO tweets (ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count, quote_count, bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term, author_username, author_name, author_verified, author_blue_verified, author_followers, author_following, sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment, collected_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (ticker_id, tweet_id, tweet_text, created_at, retweet_count, reply_count, like_count,
                            quote_count,
                            bookmark_count, lang, is_reply, is_quote, is_retweet, url, search_term,
                            author_username, author_name, author_verified, author_blue_verified,
                            author_followers, author_following, sentiment_label, sentiment_score,
                            sentiment_magnitude, weighted_sentiment, collected_at))
            self.connection.commit()
        except sqlitecloud.Error as e:
            return
            print(f"An error occurred: {e}")

    #funtion to bind the pattern and cluster data , the patterns table contains a foreign key to the clusters table
    def bind_pattern_cluster(self, stock_ID,pip_pattern_miner):
        # store these pattern in the database data.db and table patterns
        # loop through the clusters and go to the patterns table and update the cluster id
        for i, cluster in enumerate(pip_pattern_miner._pip_clusters):
            # update the Pattern_Count in the clusters table
            self.connection.execute('''
                UPDATE clusters
                SET Pattern_Count = ?
                WHERE ClusterID = ? AND StockID = ?
                ''', (len(cluster), i, stock_ID))
            
            # now loop through the patterns and update the cluster id
            for pattern in cluster:
                # update the cluster id in the patterns table
                self.connection.execute('''
                    UPDATE patterns
                    SET ClusterID = ?
                    WHERE PatternID = ? AND StockID = ?
                ''', (i, pattern, stock_ID))
                
        # commit the changes
        self.connection.commit()
        
    # function to store the prediction data in the database
    def store_prediction_data(self, stock_ID,prediction):
        # store these pattern in the database data.db and table patterns
        # get the pattern id 
        pattern_id = prediction['pattern_metrics']['pattern_id']
        # get the prediction date
        prediction_date = prediction['date']
        # get the prediction data
        prediction_Outcome = prediction
        # get confidence level
        confidence_level = prediction['confidence']
        
        res = self.connection.execute('''
            INSERT INTO Predictions (StockID, PatternID, SentimentID, PredictionDate, PredictedOutcome, ConfidenceLevel)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (stock_ID, pattern_id, 0,prediction_date, json.dumps(prediction_Outcome), confidence_level))
        
        # get the prediction id
        prediction_id = res.lastrowid
        # commit the changes
        self.connection.commit()
        
        return prediction_id
        
    # funtion to store the notification data in the database
    def store_notification_data(self, user_id, prediction_id, sent_time, notification_type, status):
        # columns are : NotificationID UserID PredictionID SentTime NotificationType Status
        # insert the data into the table
        self.connection.execute('''
            INSERT INTO Notifications (UserID, PredictionID, SentTime, NotificationType, Status)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, prediction_id, sent_time, notification_type, status))
        # commit the changes
        self.connection.commit()
        
    ##### -------- Get Functions -------- #####  
    ##### -------------------------------- #####     
    # function to get the stock data from the database
    def get_stock_data(self, stock_ID , time_frame=60):
        # get the stock data from the database
        stock_data = self.connection.execute(f'''
            SELECT * FROM stock_data WHERE StockID = {stock_ID} AND TimeFrame = {time_frame}
        ''').fetchall()
        # convert the data to a pandas dataframe
        stock_data = pd.DataFrame(stock_data, columns=['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice'])
        # convert the timestamp to datetime
        stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
        # set the timestamp as the index
        stock_data.set_index('Timestamp', inplace=True)
        # sort the data by timestamp
        stock_data.sort_index(inplace=True)
        return stock_data

    # function to get the patterns from the database
    def get_patterns_by_stock_id(self, stock_ID):
        # get the patterns from the database
        patterns = self.connection.execute(f'''
            SELECT * FROM patterns WHERE StockID = {stock_ID}
        ''').fetchall()
        # convert the data to a pandas dataframe
        patterns = pd.DataFrame(patterns, columns=['PatternID', 'StockID', 'ClusterID', 'PricePoints', 'TimeSpan', 'MarketCondition', 'Outcome', 'Label', 'MaxGain', 'MaxDrawdown'])
 
        return patterns
    def get_patterns_all(self):
        # get the patterns from the database
        patterns = self.connection.execute(f'''
            SELECT * FROM patterns
        ''').fetchall()
        # convert the data to a pandas dataframe
        patterns = pd.DataFrame(patterns, columns=['PatternID', 'StockID', 'ClusterID', 'PricePoints', 'TimeSpan', 'MarketCondition', 'Outcome', 'Label', 'MaxGain', 'MaxDrawdown'])
 
        return patterns
    
    # function to get the clusters from the database
    def get_clusters_by_stock_id(self, stock_ID):
        # get the clusters from the database
        clusters= self.connection.execute(f'''
            SELECT * FROM clusters WHERE StockID = {stock_ID}
        ''').fetchall()
        # convert the data to a pandas dataframe
        clusters = pd.DataFrame(clusters, columns=['ClusterID', 'StockID', 'AVGPricePoints', 'MarketCondition', 'Outcome', 'Label','ProbabilityScore',  'Pattern_Count', 'MaxGain', 'MaxDrawdown'])
       
        return self.filter_clusters(clusters)
    
    def get_clusters_all(self):
        # get the clusters from the database
        clusters= self.connection.execute(f'''
            SELECT * FROM clusters 
        ''').fetchall()
        # convert the data to a pandas dataframe
        clusters = pd.DataFrame(clusters, columns=['ClusterID', 'StockID', 'AVGPricePoints', 'MarketCondition', 'Outcome', 'Label','ProbabilityScore',  'Pattern_Count', 'MaxGain', 'MaxDrawdown'])
       
        return self.filter_clusters(clusters)

    # function to get to get the cluster probability score
    def get_cluster_probability_score(self,stock_id, cluster_id):
       # calculate the probability score, the probability score is if the cluster is a buy or sell pattern ,
        # then calculate tha total positive returns of it's pattern to the total returns of the patterns that belong to the cluster
        # get the patterns that belong to the cluster
        patterns = self.connection.execute(f'''
            SELECT Outcome FROM patterns WHERE ClusterID = {cluster_id} AND StockID = {stock_id}
        ''').fetchall()
        # convert the data to a pandas dataframe
        patterns = pd.DataFrame(patterns, columns=['Outcome'])
        # get the total number positive returns of the patterns that belong to the cluster
        total_positive_returns = (patterns['Outcome'] > 0).sum()
        # get the total negative returns of the patterns that belong to the cluster
        total_negative_returns = (patterns['Outcome'] < 0).sum()
        # get the total number of patterns that belong to the cluster
        total_patterns = len(patterns)
       
        # get the cluster label
        cluster_label = pd.read_sql_query(f'''
            SELECT Label FROM clusters WHERE ClusterID = {cluster_id} AND StockID = {stock_id}
        ''', self.connection)
        if cluster_label.iloc[0]['Label'] == 'Buy':
            # if the cluster is a buy pattern, then the probability score is the total positive returns of the patterns that belong to the cluster
            # divided by the total returns of the patterns that belong to the cluster
            probability_score = total_positive_returns / total_patterns
        elif cluster_label.iloc[0]['Label'] == 'Sell':
            # if the cluster is a sell pattern, then the probability score is the total negative returns of the patterns that belong to the cluster
            # divided by the total returns of the patterns that belong to the cluster
            probability_score = total_negative_returns / total_patterns
        else:
            # if the cluster is a neutral pattern, then the probability score is 0.5
            probability_score = 0.5
            
        return probability_score
    
    # funtion to get the prediction data from the database
    def get_prediction_data(self, stock_ID):
        # get the prediction data from the database
        prediction_data = self.connection.execute(f'''
            SELECT * FROM Prediction WHERE StockID = {stock_ID}
        ''').fetchall()
        # convert the data to a pandas dataframe
        prediction_data = pd.DataFrame(prediction_data, columns=['PredictionID','StockID' ,'PatternID', 'SentimentID','PredictionData' ,'PredictedOutcome' ,'ConfidenceLevel'])
        # convert the prediction data to a dictionary
        prediction_data['PredictionData'] = prediction_data['PredictionData'].apply(json.loads)
        
        return prediction_data
    
    # funtion to get the notification data from the database
    def get_notification_data(self, user_id):
        # get the notification data from the database
        notification_data = self.connection.execute(f'''
            SELECT * FROM Notifications WHERE UserID = {user_id}
        ''').fetchall()
        # convert the data to a pandas dataframe
        notification_data = pd.DataFrame(notification_data, columns=['NotificationID', 'UserID', 'PredictionID', 'SentTime', 'NotificationType', 'Status'])
        
        return notification_data
    
    # funtion to get the user email from the database
    def get_user_email(self, username):
        # get the user email from the database
        user_email = self.connection.execute(f'''
            SELECT Email FROM users WHERE Username = '{username}'
        ''').fetchall()
        # convert the data to a pandas dataframe
        #user_email = pd.DataFrame(user_email, columns=['Email'])
        
        return user_email[0][0]
    
    # function to get the user id from the database
    def get_user_id(self, username):
        # get the user id from the database
        user_id = self.connection.execute(f'''
            SELECT UserID FROM users WHERE Username = '{username}'
        ''').fetchall()
        # convert the data to a pandas dataframe
        #user_email = pd.DataFrame(user_email, columns=['Email'])
        
        return user_id[0][0]

    # function to get the news articles from the cloud database
    # def get_news_articles(self):
    #     # get the news articles from the database
    #     news_articles = self.connection.execute('''
    #         SELECT * FROM articles
    #     ''').fetchall()
    #     # convert the date to datetime
    #     news_articles['Date'] = pd.to_datetime(news_articles['Date'])
    #     # # setting the date as the index
    #     # news_articles.set_index('Date', inplace=True)
    #     # # sorting the data by date
    #     # news_articles.sort_index(inplace=True)
    #     return news_articles

    # function to get the live news articles from the database
    def get_live_news_articles(self):
        # get the live news articles from the database
        live_news_articles = self.connection.execute('''
            SELECT * FROM live_articles WHERE ID > 12153
        ''').fetchall()
        # convert the data to a pandas dataframe
        live_news_articles = pd.DataFrame(live_news_articles, columns=['ID', 'Date', 'Authors', 'Source_Domain', 'Source_Name', 'Title', 'Summary', 'Url', 'Topics', 'Ticker_Sentiment', 'Overall_Sentiment_Label', 'Overall_Sentiment_Score', 'Event_Type', 'Sentiment_Label', 'Sentiment_Score', 'Fetch_Timestamp'])
        # convert the date to datetime
        live_news_articles['Date'] = pd.to_datetime(live_news_articles['Date'])
        # # setting the date as the index
        # live_news_articles.set_index('Date', inplace=True)
        # # sorting the data by date
        # live_news_articles.sort_index(inplace=True)
        return live_news_articles

    # function to get the tweets from the database
    def get_tweets(self):
        # get the tweets from the database
        tweets = self.connection.execute('''
            SELECT * FROM tweets
        ''').fetchall()
        # convert the data to a pandas dataframe where attributes
        tweets = pd.DataFrame(tweets, columns=['ID', 'ticker_id', 'tweet_id', 'tweet_text', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count', 'lang', 'is_reply', 'is_quote', 'is_retweet', 'url', 'search_term', 'author_username', 'author_name', 'author_verified', 'author_blue_verified', 'author_followers', 'author_following', 'sentiment_label', 'sentiment_score', 'sentiment_magnitude', 'weighted_sentiment', 'collected_at'])
        # convert the date to datetime
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
        # # setting the date as the index
        # tweets.set_index('Date', inplace=True)
        # # sorting the data by date
        # tweets.sort_index(inplace=True)
        return tweets

    def get_tweets_by_date_and_ticker(self, start_date, end_date, ticker_id):
        # get the tweets from the database
        tweets = self.connection.execute('''
            SELECT * FROM tweets
            WHERE created_at BETWEEN ? AND ? AND ticker_id = ?
        ''', (start_date, end_date, ticker_id)).fetchall()
        
        # convert the data to a pandas dataframe
        tweets = pd.DataFrame(tweets, columns=['ID', 'ticker_id', 'tweet_id', 'tweet_text', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count', 'lang', 'is_reply', 'is_quote', 'is_retweet', 'url', 'search_term', 'author_username', 'author_name', 'author_verified', 'author_blue_verified', 'author_followers', 'author_following', 'sentiment_label', 'sentiment_score', 'sentiment_magnitude', 'weighted_sentiment', 'collected_at'])
        # tweets = pd.read_sql_query('''
        #     SELECT * FROM tweets
        #     WHERE created_at BETWEEN ? AND ? AND ticker_id = ?
        # ''', self.connection, params=(start_date, end_date, ticker_id))
        
        return tweets

    # function to get the tweets count by date and ticker_id
    def get_tweets_count_by_date_and_ticker(self, start_date, end_date, ticker_id):
        # get the count of tweets grouped by date and ticker_id
        tweets = self.connection.execute('''
            SELECT DATE(created_at) AS date, ticker_id, COUNT(*) AS tweet_count
            FROM tweets
            WHERE created_at BETWEEN ? AND ? AND ticker_id = ?
            GROUP BY DATE(created_at), ticker_id
            ORDER BY DATE(created_at) ASC
        ''', (start_date, end_date, ticker_id)).fetchall()
        # convert the data to a pandas dataframe
        tweets = pd.DataFrame(tweets, columns=['date', 'ticker_id', 'tweet_count'])
        # tweets = pd.read_sql_query('''
        #         SELECT
        #             DATE(created_at) AS date,
        #             ticker_id,
        #             COUNT(*) AS tweet_count
        #         FROM tweets
        #         WHERE
        #             created_at BETWEEN ? AND ?
        #             AND ticker_id = ?
        #         GROUP BY DATE(created_at), ticker_id
        #         ORDER BY DATE(created_at) ASC
        #     ''', self.connection, params=(start_date, end_date, ticker_id))
        return tweets
    ##### -------- Update Functions -------- #####  
    ##### -------------------------------- #####     
    
    # function to update the cluster probability score
    def update_cluster_probability_score_based_on_patterns(self,stock_id, cluster_id):
        probability_score = self.get_cluster_probability_score(stock_id,cluster_id)
        # update the cluster probability score
        self.connection.execute('''
            UPDATE clusters
            SET ProbabilityScore = ?
            WHERE ClusterID = ? AND StockID = ?
        ''', (probability_score, cluster_id , stock_id))
        # commit the changes
        self.connection.commit()
        return probability_score
    
    # function to update all the cluster probability score
    def update_all_cluster_probability_score(self ,stock_id, pip_pattern_miner):
        # loop through all the clusters and update the cluster probability score
        for i in range(len(pip_pattern_miner._cluster_centers)):
            self.update_cluster_probability_score_based_on_patterns(stock_id,i)

    # function to update existing tweets with sentiment data
    # def update_tweet_sentiment(self, tweet_id, sentiment_label, sentiment_score, sentiment_magnitude,
    #                            weighted_sentiment):
    #     try:
    #         cursor = self.cursor
    #         cursor.execute('''
    #             UPDATE tweets
    #             SET sentiment_label = ?,
    #                 sentiment_score = ?,
    #                 sentiment_magnitude = ?,
    #                 weighted_sentiment = ?
    #             WHERE tweet_id = ?
    #         ''', (sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment, tweet_id))
    #
    #         # Check if any rows were affected
    #         rows_affected = cursor.rowcount
    #         self.connection.commit()
    #
    #         if rows_affected == 0:
    #             print(f"Warning: No tweet found with ID {tweet_id}")
    #             return False
    #
    #         return True
    #
    #     except sqlitecloud.Error as e:
    #         print(f"Database error updating tweet {tweet_id}: {e}")
    #         return False
    # function to update existing tweets with sentiment data in the cloud using self.connection
    def update_tweet_sentiment(self, tweet_id, sentiment_label, sentiment_score, sentiment_magnitude,
                               weighted_sentiment):
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

# main function to create the database and tables
if __name__ == '__main__':
    db = Database()
    email = db.get_user_email('admin')
    print(email)
    db.close()



