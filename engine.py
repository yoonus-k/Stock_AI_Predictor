import sys
import os
from pathlib import Path

from matplotlib import pyplot as plt

# Get the project root - works in both scripts and notebooks
project_root = Path(os.getcwd())

# Add both project root and Pattern directory to path
sys.path.extend([
    str(project_root)
])

# Now import modules
from Data.db import Database

# Rest of your imports
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
from sklearn.svm import SVC

db = Database(db_name="Data/data.db")

class Engine:
    def __init__(self, lookback=24, hold_period=6):
        # var
        self.lookback = lookback +1
        self.hold_period = hold_period
        # Data
        self.data = None # The stock data
        self.clusters = None # clusters data frame
        self.model = None # SVM model
        # Database
        self.db = db
  
    # function to get the stock data
    def get_stock_data(self, stock_id):
        self.data = db.get_stock_data(stock_id)
        
    # funtion to get the clustered data
    def get_clusters(self , stock_id):
        self.clusters = db.get_clusters(stock_id)
        db.close()
        
    # function to train the SVM model
    def train_svm_model(self):
        clusters_svm = np.vstack(self.clusters['AVGPricePoints'])  
        # create labels for the clusters
        labels = np.array([i for i in range(len(clusters_svm))])
        # create the SVM model
        svm = SVC(kernel='rbf')
        # Fit the model to the training data
        svm.fit(clusters_svm, labels)
        self.model = svm
        
     # main predection function that takes the date in this format YYYY-MM-DD HH:MM:SS
     # and the stock name , then return the prediction
    def predict(self, date, stock_id):
       # Convert input date to pandas Timestamp if it isn't already
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        
        # Calculate start time
        start_time = date - pd.Timedelta(hours=self.lookback)
        
        # Filter using the datetime index
        window = self.data.loc[(self.data.index >= start_time) & (self.data.index <= date)]
        
        # if window is empty return None , the it's in weekend, then we need to get the previous data by looking back lookback *2
        if window.empty:
            # get the previous data
            start_time = date - pd.Timedelta(hours=self.lookback * 2)
            window = self.data.loc[(self.data.index >= start_time) & (self.data.index <= date)]
            
        # convert to numpy array
        window = np.array(window['ClosePrice'])

        # find pipes in the window data
        pips_x , pips_y = self.db.pip_pattern_miner.find_pips(window , 5 ,3)
        
        # reshape the pips
        pips_y = np.array(pips_y)
        
        # normalize the pips
        pips_y = scaler.fit_transform(pips_y.reshape(-1,1)).flatten()
        
        # get the current price
        current_price = window[-1]
        
        # predict using the SVM model
        cluster_prediction_indix = self.model.predict(pips_y.reshape(1, -1))
        
        # get the returns for the cluster
        cluster_return = self.clusters.iloc[cluster_prediction_indix].Outcome
        # calculate the predicted price
        predicted_price = current_price + (cluster_return * current_price)
        # get the cluster Label
        cluster_label = self.clusters.iloc[cluster_prediction_indix].Label
        # get the cluster probability
        cluster_probability = self.clusters.iloc[cluster_prediction_indix].ProbabilityScore
        # get the cluster sentiment score
        cluster_sentiment_score = self.clusters.iloc[cluster_prediction_indix].SentimentScore
        # get the cluster market condition
        cluster_market_condition = self.clusters.iloc[cluster_prediction_indix].MarketCondition
        # get the cluster MaxGain
        cluster_max_gain = self.clusters.iloc[cluster_prediction_indix].MaxGain
        # get the cluster MaxDrawdown
        cluster_max_drawdown = self.clusters.iloc[cluster_prediction_indix].MaxDrawdown
        
        # create the prediction dict
        prediction = {
            "date": date,
            "stock_id": stock_id,
            "current_price": current_price,
            "cluster_prediction_indix": cluster_prediction_indix,
            "cluster_label": cluster_label,
            "cluster_probability": cluster_probability,
            "cluster_sentiment_score": cluster_sentiment_score,
            "cluster_market_condition": cluster_market_condition,
            "cluster_return": cluster_return,
            "cluster_max_gain": cluster_max_gain,
            "cluster_max_drawdown": cluster_max_drawdown,
            "predicted_price": predicted_price,
        }
        
        # convert to data frame
        prediction = pd.DataFrame(prediction)
      
        return  prediction

    # main function to return the prediction
    def main_function(self, stock_id , date):
        # create the engine
        engine = Engine()
        
        # get the stock data
        engine.get_stock_data(stock_id)
        
        # get the clusters
        engine.get_clusters(stock_id)
        
        # train the model
        engine.train_svm_model()
        
        # predict the stock price
        prediction = engine.predict(date, stock_id)
        
        # print the prediction
        return prediction

# main if __name__ == "__main__":
if __name__ == "__main__":
    # create the engine
    engine = Engine(db)
    
    # get the stock data
    engine.get_stock_data(1)
    
    # get the clusters
    engine.get_clusters(1)
    
    # train the model
    engine.train_svm_model()
    
    # predict the stock price
    prediction = engine.predict("2025-02-21 00:00:00", 1)
    
    # print the prediction
    print(prediction)
   
       