"""
Enhanced Stock Prediction Engine
--------------------------------
This module implements an advanced prediction engine that combines pattern recognition,
sentiment analysis from news and Twitter, and reinforcement learning to generate
stock price predictions and trading recommendations.
"""

# Standard library imports
import numpy as np
import pandas as pd
import os

# Machine learning imports
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from stable_baselines3 import PPO

# Local module imports
from Data.Database.db import Database
from Data.Utils.get_stock_data import get_stock_data_from_yahoo
from Sentiment.API.alphavantage_api import get_news_sentiment_analysis
from Pattern.perceptually_important import find_pips
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TwitterAPI_Sentiment import TwitterAPI
from RL.Agents.policy import RLPolicy

# Pre-trained RL model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL", "Models", "pattern_sentiment_rl_model.zip")
model = PPO.load(model_path)

class EnhancedPredictionEngine:
    """
    Enhanced prediction engine that combines pattern recognition, sentiment analysis,
    and reinforcement learning to predict stock price movements and generate trading signals.
    """
    
    def __init__(self, lookback=24, hold_period=6, db=None):
        """
        Initialize the prediction engine with default parameters.
        
        Args:
            lookback (int): Number of time periods to look back for pattern recognition
            hold_period (int): Number of time periods to hold a position
            db (Database, optional): Database connection, creates new one if None
        """
        # Initialize RL policy and model with correct path
        rl_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "RL", "Models", "pattern_sentiment_rl_model.zip")
        self.rl_policy = RLPolicy(rl_model_path)
        
        # Core parameters
        self.lookback = lookback + 1
        self.hold_period = hold_period
        
        # Data storage
        self.data = None
        self.clusters = None
        self.cluster_centers = None
        self.model = None
        
        # Initialize components
        self.db = db if db else Database()
        self.scaler = MinMaxScaler()
        self.twitter_api = TwitterAPI(self.db)
        
        # Sentiment weights configuration
        self.sentiment_weights = {
            'twitter': 0.5,
            'news': 0.5,
            'pattern_vs_sentiment': 0.6  # Pattern weight vs sentiment weight
        }
        
        # Supported stocks dictionary
        self.stocks = {
            1: "GOLD (XAUUSD)",
            2: "BTC (BTCUSD)",
            3: "APPL (AAPL)",
            4: "Amazon (AMZN)",
            5: "NVIDIA (NVDA)",
        }
    
    #---------------------------------------------------------------------------
    # Risk analysis methods
    #---------------------------------------------------------------------------
    
    def calculate_reward_risk_ratio(self):
        """
        Calculate the maximum reward to risk ratio across all clusters.
        
        Returns:
            float: Maximum reward-to-risk ratio
        """
        max_reward_ratio = 0
        # Loop through the clusters and calculate the reward to risk ratio
        for index, row in self.clusters.iterrows():
            max_gain = row['MaxGain']
            max_drawdown = row['MaxDrawdown']
            # Add small epsilon to avoid division by zero
            reward_risk_ratio = max_gain / (abs(max_drawdown) + 0.00000000000001)
            if reward_risk_ratio > max_reward_ratio:
                max_reward_ratio = reward_risk_ratio
        return max_reward_ratio
    
    def calculate_average_reward_risk_ratio(self):
        """
        Calculate the average reward to risk ratio across all clusters.
        
        Returns:
            float: Average reward-to-risk ratio
        """
        reward_risk_ratios = []
        # Loop through the clusters and calculate the reward to risk ratio
        for index, row in self.clusters.iterrows():
            max_gain = row['MaxGain']
            max_drawdown = row['MaxDrawdown']
            # Add small epsilon to avoid division by zero
            reward_risk_ratio = max_gain / (abs(max_drawdown) + 0.00000000000001)
            if reward_risk_ratio > 0:
                reward_risk_ratios.append(reward_risk_ratio)
        
        # Avoid division by zero if no positive ratios
        if not reward_risk_ratios:
            return 0
            
        average_reward_risk_ratio = sum(reward_risk_ratios) / len(reward_risk_ratios)
        return average_reward_risk_ratio
    
    def calculate_position_size(self, confidence, risk_percentage, risk_to_reward_ratio, account_balance):
        """
        Calculate the position size based on risk parameters and confidence.
        
        Args:
            confidence (float): Confidence score (0-1)
            risk_percentage (float): Percentage of account willing to risk (0-1)
            risk_to_reward_ratio (float): Ratio of potential reward to risk
            account_balance (float): Total account balance
            
        Returns:
            float: Recommended position size
        """
        # Calculate the risk amount
        risk_amount = account_balance * risk_percentage 
        # Calculate the position size adjusted by confidence
        position_size = risk_amount * risk_to_reward_ratio * (1 + confidence)
        return position_size
    
    #---------------------------------------------------------------------------
    # Data retrieval and preparation methods
    #---------------------------------------------------------------------------
    
    def get_stock_data(self, stock_id, date, period=5):
        """
        Retrieve stock data from database or external source if not available.
        
        Args:
            stock_id (int): ID of the stock to retrieve
            date (str): Date to retrieve data for
            period (int): Number of days of historical data to retrieve
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        # Try to get data from database first
        stock_data = self.db.get_stock_data_by_date_and_period(stock_id, date, period)
        
        # Check if the stock data is available in the database for the given date or 2 days before
        df_last_date = stock_data.index[-1] if not stock_data.empty else None
        
        self.data = stock_data
        
        is_data_available = df_last_date and (df_last_date >= pd.to_datetime(date) - pd.Timedelta(days=1))
        
        # If data not available, fetch from external source
        if not is_data_available:
            print(f"No data found for stock ID {stock_id}.")
            print("Fetching data from Yahoo Finance...")
            stock_data = get_stock_data_from_yahoo(stock_id, date, time_frame='60m', period=7)
            if stock_data.empty:
                raise ValueError(f"No data found for stock id {stock_id}.")
            self.data = stock_data
            
        return self.data
    
    def get_clusters(self, stock_id):
        """
        Retrieve cluster data for the specified stock from database.
        
        Args:
            stock_id (int): ID of the stock to retrieve clusters for
        """
        self.clusters = self.db.get_clusters_by_stock_id(stock_id)
        
        # Extract features and labels from clusters
        features = self.clusters['AVGPricePoints'].values
        
        # Convert the features to a 2D array
        features = np.array([np.array(x.split(','), dtype=float) for x in features])
        
        self.cluster_centers = features
    
    #---------------------------------------------------------------------------
    # Model training and prediction methods
    #---------------------------------------------------------------------------
    
    def train_svm_model(self):
        """
        Train an SVM model on the cluster centers for pattern classification.
        """
        labels = np.array([i for i in range(len(self.cluster_centers))])
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(self.cluster_centers, labels)
        self.model = svm
    
    def apply_decision_matrix(self, pattern_label, pattern_confidence, sentiment_score):
        """
        Apply a decision matrix to determine action and confidence based on 
        pattern and sentiment analysis.
        
        Args:
            pattern_label (str): Label of the identified pattern
            pattern_confidence (float): Confidence score for the pattern
            sentiment_score (float): Sentiment score
            
        Returns:
            tuple: (action recommendation, confidence adjustment)
        """
        # Dynamic configuration for thresholds
        thresholds = {
            'confidence': {
                'high': 0.6,
                'medium': 0.50,
            },
            'sentiment': {
                'strong_positive': 0.5,
                'positive': 0.2,
                'neutral_min': -0.2,
                'negative': -0.2,
                'strong_negative': -0.5,
            },
            'confidence_boosts': {
                'strong': 0.30,
                'moderate': 0.20,
                'light': 0.10,
                'weak': 0.05,
                'none': 0.00,
                'penalty_light': -0.05,
                'penalty_moderate': -0.10,
                'penalty_strong': -0.20,
            }
        }

        label = pattern_label.upper()

        # Logic for BUY patterns
        if "BUY" in label:
            if sentiment_score >= thresholds['sentiment']['strong_positive']:
                if pattern_confidence >= thresholds['confidence']['high']:
                    return "STRONG BUY", thresholds['confidence_boosts']['strong']
                else:
                    return "BUY", thresholds['confidence_boosts']['moderate']
            elif sentiment_score >= thresholds['sentiment']['positive']:
                return "BUY", thresholds['confidence_boosts']['light']
            elif sentiment_score <= thresholds['sentiment']['strong_negative']:
                return "HOLD", thresholds['confidence_boosts']['penalty_moderate']
            else:
                return "WEAK BUY", thresholds['confidence_boosts']['weak']

        # Logic for SELL patterns
        elif "SELL" in label:
            if sentiment_score <= thresholds['sentiment']['strong_negative']:
                if pattern_confidence >= thresholds['confidence']['high']:
                    return "STRONG SELL", -thresholds['confidence_boosts']['strong']
                else:
                    return "SELL", -thresholds['confidence_boosts']['moderate']
            elif sentiment_score <= thresholds['sentiment']['negative']:
                return "SELL", -thresholds['confidence_boosts']['light']
            elif sentiment_score >= thresholds['sentiment']['strong_positive']:
                return "HOLD", thresholds['confidence_boosts']['light']
            else:
                return "WEAK SELL", -thresholds['confidence_boosts']['weak']

        # Logic for neutral patterns
        else:
            if sentiment_score >= thresholds['sentiment']['strong_positive']:
                return "BUY", thresholds['confidence_boosts']['light']
            elif sentiment_score <= thresholds['sentiment']['strong_negative']:
                return "SELL", -thresholds['confidence_boosts']['light']
            else:
                return "HOLD", thresholds['confidence_boosts']['none']
    
    def sentiment_adjusted_prediction(self, pattern_pred, current_price, sentiment, sensitivity=0.002):
        """
        Adjust price prediction based on sentiment analysis.
        
        Args:
            pattern_pred (float): Price prediction from pattern analysis
            current_price (float): Current price
            sentiment (dict): Sentiment data dictionary
            sensitivity (float): Sensitivity parameter for adjustment
            
        Returns:
            float: Sentiment-adjusted price prediction
        """
        sentiment_multiplier = np.tanh(sentiment['impact_score'] * 2)
        adjustment = current_price * sensitivity * sentiment_multiplier
        return pattern_pred + adjustment
    
    def predict(self, stock_id, date):
        """
        Generate a complete prediction for a stock at a specific date.
        
        Args:
            stock_id (int): ID of the stock to predict
            date (str): Date to generate prediction for
            
        Returns:
            dict: Complete prediction data including pattern analysis, 
                  sentiment analysis, and final recommendation
        """
        # Get the stock symbol and ticker
        symbol = self.stocks[stock_id]
        ticker = symbol.split('(')[-1].replace(')', '').strip()
       
        # Convert and validate date
        date_datetime = pd.to_datetime(date)
        start_time = date_datetime - pd.Timedelta(days=self.lookback)
        window = self.data.loc[(self.data.index >= start_time) & (self.data.index <= date_datetime)]
        
        #-------------------
        # Pattern detection
        #-------------------
        window_prices = np.array(window['ClosePrice'])
        
        # Convert to 1D array
        window_prices = window_prices.flatten()
        
        # Get only the last lookback values
        if len(window_prices) > self.lookback:
            window_prices = window_prices[:self.lookback]
            # Reverse the order of the window prices to match the database format
            window_prices = window_prices[::-1]
        
        # Find perceptually important points (PIPs)
        pips_x, pips_y = find_pips(window_prices, 5, 3)
        pips_y = self.scaler.fit_transform(np.array(pips_y).reshape(-1, 1)).flatten()
        
        # Pattern prediction
        current_price = window_prices[-1]
        cluster_idx = self.model.predict(pips_y.reshape(1, -1))[0]
        
        # Check the mean squared error of the model
        mse = mean_squared_error(pips_y, self.cluster_centers[cluster_idx])
        
        # Optional MSE-based filtering
        # Uncomment to filter out predictions with high error
        # if mse > 0.03:
        #     print(f"Mean Squared Error: {mse:.4f} - Prediction is not reliable.")
        #     return None

        #-------------------
        # Get cluster data
        #-------------------
        cluster_data = self.clusters.iloc[cluster_idx]
        pattern_return = cluster_data['Outcome']
        predicted_price = current_price * (1 + pattern_return)
        cluster_label = cluster_data['Label']
        cluster_probs = cluster_data['ProbabilityScore']
        cluster_market_condition = cluster_data['MarketCondition']
        cluster_max_gain = cluster_data['MaxGain']
        cluster_max_drawdown = cluster_data['MaxDrawdown']
        reward_risk_ratio = cluster_max_gain / abs(cluster_max_drawdown)

        #-------------------
        # Sentiment analysis
        #-------------------
        # Get news sentiment data
        sentiment_data_all = get_news_sentiment_analysis(ticker, date)
        
        # Get Twitter sentiment data
        twitter_sentiment = self.twitter_api.get_tweets_sentiment_analysis(ticker_id=stock_id, specific_date=date)
        # Extract sentiment scores
        sentiment_score = sentiment_data_all['Predicted News Sentiment Score']
        impact_score = sentiment_data_all['Predicted Impact Score']
        
        # Create consolidated sentiment data
        sentiment_data = {
            'twitter_score': twitter_sentiment['tweets_sentiment_score'],
            'news_score': sentiment_score,
            'impact_score': impact_score,
        }
        
        #-------------------
        # Generate hybrid prediction
        #-------------------
        # Adjust prediction based on sentiment
        final_price = self.sentiment_adjusted_prediction(
            predicted_price, current_price, sentiment_data
        )
        
        # Decision matrix approach - currently disabled
        # action, confidence_boost = self.apply_decision_matrix(
        #     pattern_label=cluster_data['Label'],
        #     pattern_confidence=cluster_probs,
        #     sentiment_score=sentiment_data['impact_score'],
        # )
        
        #-------------------
        # RL-based decision
        #-------------------
        # Construct RL state
        rl_state = self.rl_policy.build_state(
            pattern_data={
                'probability': cluster_probs,
                'action': cluster_label,
                'reward_risk_ratio': reward_risk_ratio,
                'max_gain': cluster_max_gain,
                'max_drawdown': cluster_max_drawdown,
            },
            sentiment_data={
                'impact_score': impact_score,
                'news_score': sentiment_score,
                'twitter_score': twitter_sentiment['tweets_sentiment_score'],
            },
        )

        # Get action and confidence from RL model
        action = self.rl_policy.select_action(rl_state)
        confidence_boost = self.rl_policy.estimate_confidence(rl_state)
        print(f"Action: {action}, Confidence Boost: {confidence_boost:.2f}")
        
        # Calculate overall confidence
        confidence_all = min(0.99, cluster_probs + confidence_boost)
        
        # Calculate position size
        risk_percentage = 0.05
        position_size = self.calculate_position_size(
            confidence_all, risk_percentage, reward_risk_ratio, 100000
        )
       
        #-------------------
        # Compile results
        #-------------------
        return {
            'date': date,
            'stock_id': stock_id,
            'stock_name': symbol,
            'current_price': current_price,
            'pattern_prediction': predicted_price,
            'final_prediction': final_price,
            'confidence': confidence_all,
            'action': action,
            'position_size': position_size,
            'pattern_metrics': {
                'pattern_id': cluster_idx,
                'type': cluster_label,
                'probability': cluster_probs,
                'max_gain': cluster_max_gain,
                'max_drawdown': cluster_max_drawdown,
                'reward_risk_ratio': reward_risk_ratio,
            },
            'sentiment_metrics': sentiment_data_all,
            'twitter_sentiment': twitter_sentiment,  
        }
    
    def generate_report(self, prediction):
        """
        Generate a formatted prediction report with all information.
        
        Args:
            prediction (dict): Prediction data dictionary
            
        Returns:
            str: Formatted report
        """
        report = f"""
        === Enhanced Prediction Report ===
        Stock ID: {prediction['stock_id']}
        Stock Name: {prediction['stock_name']}
        Date And Time: {prediction['date']}
        Current Price: ${prediction['current_price']:.2f}

        --- Pattern Analysis ---
        ID: {prediction['pattern_metrics']['pattern_id']}
        Type: {prediction['pattern_metrics']['type']}
        Predicted Price: ${prediction['pattern_prediction']:.2f}
        Confidence: {prediction['pattern_metrics']['probability']:.1%}
        Max Gain: {prediction['pattern_metrics']['max_gain']:.2%}
        Max Drawdown: {prediction['pattern_metrics']['max_drawdown']:.2%}
        Reward/Risk Ratio: {prediction['pattern_metrics']['reward_risk_ratio']:.2f}
        Position Size: ${prediction['position_size']:.2f}

        --- Sentiment Analysis ---
        News Score: {prediction['sentiment_metrics']['Predicted News Sentiment Score']:.2f}
        Impact Score: {prediction['sentiment_metrics']['Predicted Impact Score']:.2f}
        Most Relevant Article summary: {prediction['sentiment_metrics']['Most Relevant Article']['summary'] if prediction['sentiment_metrics']['Most Relevant Article'] else 'No relevant article found'}
        Most Relevant Article Title: {prediction['sentiment_metrics']['Most Relevant Article']['title'] if prediction['sentiment_metrics']['Most Relevant Article'] else 'No relevant article found'}
        Most Relevant Article URL: {prediction['sentiment_metrics']['Most Relevant Article']['url'] if prediction['sentiment_metrics']['Most Relevant Article'] else 'No relevant article found'} 
        
        --- Twitter Sentiment Analysis ---
        Twitter Sentiment Score: {prediction['twitter_sentiment']['tweets_sentiment_score']:.2f}
        Twitter Sentiment Count: {prediction['twitter_sentiment']['tweets_count']}
        Twitter Most Positive Tweet: {prediction['twitter_sentiment']['most_positive_tweet']}
        Twitter Most Negative Tweet: {prediction['twitter_sentiment']['most_negative_tweet']}
        Twitter Weighted Sentiment Score: {prediction['twitter_sentiment']['tweets_weighted_sentiment_score']:.2f}

        --- Final Recommendation ---
        Predicted Price: ${prediction['final_prediction']:.2f}
        Action: {prediction['action']}
        Confidence: {prediction['confidence']:.1%}
        """
        return report

    def main_function(self, stock_id, date):
        """
        Main function to run the full prediction pipeline.
        
        Args:
            stock_id (int): ID of the stock to predict
            date (str): Date to generate prediction for
            
        Returns:
            tuple: (formatted report, prediction data)
        """
        engine = EnhancedPredictionEngine()
        engine.get_stock_data(stock_id, date)
        engine.get_clusters(stock_id)
        engine.train_svm_model()
        
        prediction = engine.predict(stock_id, date)
        
        # Format the prediction
        prediction_report = engine.generate_report(prediction)
        
        return prediction_report, prediction


# Run this script directly
if __name__ == "__main__":
    engine = EnhancedPredictionEngine()
    engine.get_stock_data(3, "2025-04-10", 5)
    
    engine.get_clusters(3)
    engine.train_svm_model()
    
    prediction = engine.predict(3, "2025-04-10")
    
    # Format the prediction
    # prediction_report = engine.generate_report(prediction)
    # print(prediction)
