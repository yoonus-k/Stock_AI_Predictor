import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from matplotlib import pyplot as plt
from Data.db_cloud import Database
from Sentiment.alphavantage_api import get_news_sentiment_analysis
from Pattern.perceptually_important import find_pips

class EnhancedPredictionEngine:
    def __init__(self, lookback=24, hold_period=6):
        self.lookback = lookback + 1
        self.hold_period = hold_period
        self.data = None
        self.clusters = None
        self.model = None
        self.db = Database()
        self.scaler = MinMaxScaler()
        
        # Sentiment weights configuration
        self.sentiment_weights = {
            'twitter': 0.7,
            'news': 0.3,
            'pattern_vs_sentiment': 0.6  # Pattern weight vs sentiment weight
        }
        
        self.stocks ={
        1: "GOLD (XAUUSD)",
        2: "BTC (BTCUSD)",
        3: "APPL (AAPL)",
        4: "Amazon (AMZN)",
        5: "NVIDIA (NVDA)",
        }
        
    # funtion to calculate the maximum reward to risk ratio
    def calculate_reward_risk_ratio(self):
        max_reward_ratio = 0
        # loop through the clusters and calculate the reward to risk ratio
        for index, row in self.clusters.iterrows():
            max_gain = row['MaxGain']
            max_drawdown = row['MaxDrawdown']
            reward_risk_ratio = max_gain / (abs(max_drawdown)+0.00000000000001)
            if reward_risk_ratio > max_reward_ratio:
                max_reward_ratio = reward_risk_ratio
        return max_reward_ratio
    # funtion to calculate the average reward to risk ratio
    def calculate_average_reward_risk_ratio(self):
        reward_risk_ratios = []
        # loop through the clusters and calculate the reward to risk ratio
        for index, row in self.clusters.iterrows():
            max_gain = row['MaxGain']
            max_drawdown = row['MaxDrawdown']
            reward_risk_ratio = max_gain / (abs(max_drawdown)+0.00000000000001)
            if reward_risk_ratio > 0:
                reward_risk_ratios.append(reward_risk_ratio)
        average_reward_risk_ratio = sum(reward_risk_ratios)/len(reward_risk_ratios)
        return average_reward_risk_ratio
    
    
    def get_stock_data(self, stock_id):
        self.data = self.db.get_stock_data(stock_id)
        
    def get_clusters(self, stock_id):
        self.clusters = self.db.get_clusters(stock_id)
        
    def train_svm_model(self):
        # Extract features and labels from clusters
        features = self.clusters['AVGPricePoints'].values
        # convert the features to a 2D array
        features = np.array([np.array(x.split(','), dtype=float) for x in features])
        labels = np.array([i for i in range(len(features))])
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(features, labels)
        self.model = svm
        
    def calculate_hybrid_score(self, pattern_data, sentiment_data):
        """Calculate weighted score combining pattern and sentiment"""
        pattern_strength = self.sentiment_weights['pattern_vs_sentiment'] * (
            pattern_data['probability'] * pattern_data['reward_risk_ratio']
        ) # expected return 
        
        sentiment_strength = (1 - self.sentiment_weights['pattern_vs_sentiment']) * (
            self.sentiment_weights['twitter'] * sentiment_data['twitter_score'] + 
            self.sentiment_weights['news'] * sentiment_data['news_score']
        )
        
        return pattern_strength + sentiment_strength
    
    def apply_decision_matrix(self, pattern_label,pattern_confidence, sentiment_score):
            """Enhanced decision matrix combining pattern labels and sentiment"""
            # Updated thresholds
            decision_thresholds = {
                'high_confidence': 0.75,
                'medium_confidence': 0.45,
                'strong_positive_sentiment': 0.5,
                'positive_sentiment': 0.3,
                'negative_sentiment': -0.3,
                'strong_negative_sentiment': -0.5
            }

            # Normalize pattern label
            pattern_direction = pattern_label.upper()
            
            # Decision logic
            if "BUY" in pattern_direction:
                if sentiment_score >= decision_thresholds['strong_positive_sentiment']:
                    if pattern_confidence >= decision_thresholds['high_confidence']:
                        return 'STRONG BUY', 0.3
                    else:
                        return 'BUY', 0.2
                elif sentiment_score >= decision_thresholds['positive_sentiment']:
                    return 'BUY', 0.15
                elif sentiment_score <= decision_thresholds['strong_negative_sentiment']:
                    return 'HOLD', -0.1
                else:  # Neutral sentiment
                    return 'WEAK BUY', 0.05

            elif "SELL" in pattern_direction:
                if sentiment_score <= decision_thresholds['strong_negative_sentiment']:
                    if pattern_confidence >= decision_thresholds['high_confidence']:
                        return 'STRONG SELL', -0.3
                    else:
                        return 'SELL', -0.2
                elif sentiment_score <= decision_thresholds['negative_sentiment']:
                    return 'SELL', -0.15
                elif sentiment_score >= decision_thresholds['strong_positive_sentiment']:
                    return 'HOLD', 0.1
                else:  # Neutral sentiment
                    return 'WEAK SELL', -0.05

            else:  # Neutral patterns
                if sentiment_score >= decision_thresholds['strong_positive_sentiment']:
                    return 'BUY', 0.1
                elif sentiment_score <= decision_thresholds['strong_negative_sentiment']:
                    return 'SELL', -0.1
                else:
                    return 'HOLD', 0.0
        
    def sentiment_adjusted_prediction(self, pattern_pred, current_price, sentiment,sensitivity=0.002):
        """Adjust prediction based on sentiment analysis"""
        sentiment_multiplier = np.tanh(sentiment['impact_score'] * 2)
        adjustment = current_price * sensitivity * sentiment_multiplier
        return pattern_pred + adjustment
    
    def calculate_position_size(self, current_price, hybrid_score, max_drawdown):
        """Dynamic position sizing based on confidence"""
        base_size = 1000  # USD
        risk_adjusted_size = base_size * (1 + hybrid_score)
        
        # Apply max drawdown cap (no more than 2% risk per trade)
        max_risk_size = current_price * 0.02 / abs(max_drawdown)
        return min(5000, max(200, risk_adjusted_size, max_risk_size))
    
    def predict(self,stock_id, date):
        # get the companies dictionary from the class
        symbol = self.stocks[stock_id]
        # Extract the ticker symbol
        ticker = symbol.split('(')[-1].replace(')', '').strip()
        
        # Convert and validate date
        date = pd.to_datetime(date)
        start_time = date - pd.Timedelta(hours=self.lookback)
        window = self.data.loc[(self.data.index >= start_time) & (self.data.index <= date)]
        
        if window.empty:
            start_time = date - pd.Timedelta(hours=self.lookback * 2)
            window = self.data.loc[(self.data.index >= start_time) & (self.data.index <= date)]
            
        # Pattern detection
        window_prices = np.array(window['ClosePrice'])
        pips_x, pips_y = find_pips(window_prices, 5, 3)
        pips_y = self.scaler.fit_transform(np.array(pips_y).reshape(-1, 1)).flatten()
        
        # Pattern prediction
        current_price = window_prices[-1]
        cluster_idx = self.model.predict(pips_y.reshape(1, -1))[0]
        
        
        # Get cluster data
        cluster_data = self.clusters.iloc[cluster_idx]
        pattern_return = cluster_data['Outcome']
        predicted_price = current_price * (1 + pattern_return)
        cluster_label = cluster_data['Label']
        cluster_probs = cluster_data['ProbabilityScore']
        cluster_market_condition = cluster_data['MarketCondition']
        cluster_max_gain = cluster_data['MaxGain']
        cluster_max_drawdown = cluster_data['MaxDrawdown']
        reward_risk_ratio = cluster_max_gain / abs(cluster_max_drawdown)

        
        # get the sentiment data
        sentiment_data = self.get_news_sentiment(ticker, date)
        
        # extract the sentiment score
        sentiment_score = sentiment_data['Predicted News Sentiment Score']
        # get the sentiment impact score
        impact_score = sentiment_data['Predicted Impact Score']
        
        # create the pattern data
        pattern_data = {
            'probability': cluster_probs,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'max_gain': cluster_max_gain,
            'max_drawdown': cluster_max_drawdown,
            'reward_risk_ratio': reward_risk_ratio,
        }
        
        # create the sentiment data
        sentiment_data = {
            'twitter_score': 0.5,  # Example score
            'news_score': sentiment_score,
            'impact_score': impact_score,
        }
        
        # Generate hybrid prediction
        # Combined analysis
        hybrid_score = self.calculate_hybrid_score(pattern_data, sentiment_data)
        final_price = self.sentiment_adjusted_prediction(
            predicted_price, current_price, sentiment_data
        )
        action, confidence_boost = self.apply_decision_matrix(
            pattern_label=cluster_data['Label'],
            pattern_confidence=cluster_probs,
            sentiment_score=sentiment_data['impact_score'],
        )
        position_size = self.calculate_position_size(
            current_price, hybrid_score, cluster_max_drawdown
        )
        
        # Compile results
        return {
            'date': date,
            'stock_id': stock_id,
            'stock_name': symbol,
            'current_price': current_price,
            'pattern_prediction': predicted_price,
            'final_prediction': final_price,
            'confidence': min(0.99, cluster_probs + confidence_boost),
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
            'sentiment_metrics': sentiment_data,
            'hybrid_score': hybrid_score
        }
    
    def get_news_sentiment(self, ticker, date):
        """Fetch news sentiment analysis for the stock"""
        news_sentiment = get_news_sentiment_analysis( ticker, date)
        return news_sentiment
        
    def generate_report(self, prediction):
        """Generate formatted prediction report with all information"""
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

        --- Sentiment Analysis ---
        Twitter Score: {prediction['sentiment_metrics']['twitter_score']:.2f}
        News Score: {prediction['sentiment_metrics']['news_score']:.2f}
        Impact Score: {prediction['sentiment_metrics']['impact_score']:.2f}

        --- Hybrid Analysis ---
        Hybrid Score: {prediction['hybrid_score']:.2f}

        --- Final Recommendation ---
        Predicted Price: ${prediction['final_prediction']:.2f}
        Action: {prediction['action']}
        Confidence: {prediction['confidence']:.1%}
        Position Size: ${prediction['position_size']:.2f}
        """
        return report


    # main funtion 
    def main_function(self,stock_id, date):
        engine = EnhancedPredictionEngine()
        engine.get_stock_data(stock_id)
        engine.get_clusters(stock_id)
        engine.train_svm_model()
        
        prediction = engine.predict(stock_id, date)
        
        # format the prediction
        prediction_report = engine.generate_report(prediction)
        
        print(prediction_report)
    

if __name__ == "__main__":
    engine = EnhancedPredictionEngine()
    engine.get_stock_data(3)
    engine.get_clusters(3)
    engine.train_svm_model()
    
    prediction = engine.predict(3,"2025-04-10 16:00:00")
    
    # format the prediction
    prediction_report = engine.generate_report(prediction)
    print(prediction_report)
    
    
    # engine = EnhancedPredictionEngine()
    # engine.get_clusters(1)
    # print(engine.calculate_reward_risk_ratio())
    # print(engine.calculate_average_reward_risk_ratio())