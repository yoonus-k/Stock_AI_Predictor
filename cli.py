import random
import datetime
from engine import Engine
pattern_engine = Engine()
# Simulated company stock list
companies = {
    1: "GOLD (GLD)",
    2: "BTC (BTC)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

def get_stock_prediction(stock_id, date):
    
    prediction= pattern_engine.main_function(stock_id, date)
    current_price = prediction["current_price"].item()
    predicted_price = prediction["predicted_price"].item()
    pattern_id = prediction["cluster_prediction_indix"].item()
    pattern_label = prediction["cluster_label"].item()
    pattern_probability = prediction["cluster_probability"].item()
    pattern_max_gain = prediction["cluster_max_gain"].item()
    pattern_max_drawdown = prediction["cluster_max_drawdown"].item()
    

    # Generate sentiment scores
    twitter_sentiment = round(random.uniform(-1, 1), 2)  # Sentiment score between -1 to 1
    news_sentiment = round(random.uniform(-1, 1), 2)  # Sentiment score between -1 to 1
    impact_score = round((twitter_sentiment + news_sentiment) / 2, 2)  # Average sentiment impact

    # Simulated news summary
    news_summaries = [
        "Tech stocks surge after positive earnings reports.",
        "Market volatility increases due to global economic uncertainty.",
        "Investors remain optimistic despite recent market downturns.",
        "Stock prices are expected to fluctuate due to recent policy changes.",
        "Analysts predict stable growth for the tech sector in Q2."
    ]
    news_summary = random.choice(news_summaries)

    return current_price,predicted_price, pattern_id, pattern_label, pattern_probability,pattern_max_gain, pattern_max_drawdown, twitter_sentiment, news_sentiment, impact_score, news_summary

# CLI Simulation
def main():
    print("\n=== Stock Market Prediction CLI ===\n")
    
    # log in to the system , ask for username and password
    while True:
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        if pattern_engine.db.login(username, password):
            print("Login successful!")
            break
        else:
            print("Invalid username or password. Please try again.")
    
    
    # Display available stocks
    print("Select a stock to predict:")
    for key, value in companies.items():
        print(f"{key}. {value}")
    
    # Get user input for stock selection
    while True:
        try:
            stock_choice = int(input("\nEnter stock number: "))
            if stock_choice in companies:
                stock_name = companies[stock_choice]
                stock_id = stock_choice
                break
            else:
                print("Invalid choice. Please select a valid stock number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get user input for date selection
    while True:
        try:
            date_input = input("\nEnter the date for prediction (YYYY-MM-DD): ")
            prediction_date = datetime.datetime.strptime(date_input, "%Y-%m-%d").date()
            break
        except ValueError:
            print("Invalid date format. Please enter in YYYY-MM-DD format.")

    print("\nFetching real-time stock and sentiment data...\n")

    # Simulate prediction results
    current_price,predicted_price, pattern_id, pattern_label, pattern_probability,pattern_max_gain, pattern_max_drawdown, twitter_sentiment, news_sentiment, impact_score, news_summary = get_stock_prediction(stock_id, prediction_date)

    # Display the results
    print("=== Prediction Results ===")
    print(f"Stock Selected: {stock_name}")
    print(f"Prediction Date: {prediction_date}")
    print(f"Current Price: ${current_price}")
    print(f"Predicted Next Price: ${predicted_price}")
    print(f"Pattern ID: {pattern_id}")
    print(f"Pattern Label: {pattern_label}")
    print(f"Pattern Probability: {pattern_probability}%")
    print(f"Pattern Maximum Gain (Reward): {pattern_max_gain}%")
    print(f"Pattern Maximum Drawdown (Risk): {pattern_max_drawdown}%")
    print(f"Predicted Twitter Sentiment Score: {twitter_sentiment}")
    print(f"Predicted News Sentiment Score: {news_sentiment}")
    print(f"Predicted Impact Score: {impact_score}")
    print(f"Summary of the News: {news_summary}")
    print("\nThank you for using the Stock Market Prediction CLI!")

if __name__ == "__main__":
    main()
