# this model is used to connect to the database and create the tables also stores the data
import sys
import os
from pathlib import Path
import numpy as np
import bcrypt
import json
# Get the current working directory (where the notebook/script is running)
current_dir = Path(os.getcwd())
# Navigate to the 'main' folder (adjust if needed)
main_dir = str(current_dir.parent)  # If notebook is inside 'main'
# OR if notebook is outside 'main':
# main_dir = str(current_dir / "main")  # Assumes 'main' is a subfolder
sys.path.append(main_dir)
import pandas as pd
import sqlitecloud
# get the sqlite url from the .env file
from dotenv import load_dotenv

companies = {
    1: "GOLD (XAUUSD)",
    2: "BTC (BTCUSD)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}


class Database:
    def __init__(self):
        load_dotenv()
        self.sqlitecloud_url = os.getenv('SQLITECLOUD_URL')
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Connect to the SQLite database."""
        self.connection = sqlitecloud.connect(self.sqlitecloud_url)
        self.cursor = self.connection.cursor()
        print(f"Connected to sqlite cloud database")

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
        
    ##### -------- Get Functions -------- #####  
    ##### -------------------------------- #####     
    # function to get the stock data from the database
    def get_stock_data(self, stock_ID , time_frame=60):
        # get the stock data from the database
        stock_data = self.connection.execute(f'''
            SELECT * FROM stock_data WHERE StockID = {stock_ID} AND TimeFrame = {time_frame}
        ''').fetchall()
        # convert the data to a pandas dataframe
        stock_data = pd.DataFrame(stock_data, columns=['StockEntryID', 'StockID', 'StockSymbol', 'Timestamp', 'TimeFrame', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'Volume'])
        # convert the timestamp to datetime
        stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
        # set the timestamp as the index
        stock_data.set_index('Timestamp', inplace=True)
        # sort the data by timestamp
        stock_data.sort_index(inplace=True)
        return stock_data

    # function to get the patterns from the database
    def get_patterns(self, stock_ID):
        # get the patterns from the database
        patterns = self.connection.execute(f'''
            SELECT * FROM patterns WHERE StockID = {stock_ID}
        ''').fetchall()
        # convert the data to a pandas dataframe
        patterns = pd.DataFrame(patterns, columns=['PatternID', 'StockID', 'ClusterID', 'PricePoints', 'TimeSpan', 'MarketCondition', 'Outcome', 'Label', 'MaxGain', 'MaxDrawdown'])
 
        return patterns
    
    # function to get the clusters from the database
    def get_clusters(self, stock_ID):
        # get the clusters from the database
        clusters= self.connection.execute(f'''
            SELECT * FROM clusters WHERE StockID = {stock_ID}
        ''').fetchall()
        # convert the data to a pandas dataframe
        clusters = pd.DataFrame(clusters, columns=['ClusterID', 'StockID', 'AVGPricePoints', 'MarketCondition', 'Outcome', 'Label','ProbabilityScore',  'Pattern_Count', 'MaxGain', 'MaxDrawdown'])
       
        return clusters

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
    
    # funtion to update all the cluster probability score
    def update_all_cluster_probability_score(self ,stock_id, pip_pattern_miner):
        # loop through all the clusters and update the cluster probability score
        for i in range(len(pip_pattern_miner._cluster_centers)):
            self.update_cluster_probability_score_based_on_patterns(stock_id,i)
        
# main function to create the database and tables
if __name__ == '__main__':
    db = Database()
    df= db.get_notification_data(1)
    print(df)
    db.close()



