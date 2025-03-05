import random
import datetime

# Simulated company stock list
companies = {
    1: "Apple (AAPL)",
    2: "Microsoft (MSFT)",
    3: "Tesla (TSLA)",
    4: "Amazon (AMZN)",
    5: "Google (GOOGL)",
}

# Simulated pattern data
patterns = {
    101: {"label": "Bullish", "probability": round(random.uniform(70, 95), 2), "max_drawdown": round(random.uniform(5, 15), 2)},
    102: {"label": "Bearish", "probability": round(random.uniform(60, 90), 2), "max_drawdown": round(random.uniform(10, 20), 2)},
    103: {"label": "Neutral", "probability": round(random.uniform(50, 80), 2), "max_drawdown": round(random.uniform(3, 10), 2)}
}

# Simulated function to get predictions (randomized for demo purposes)
def get_stock_prediction(stock, date):
    predicted_price = round(random.uniform(100, 500), 2)  # Simulated price
    
    # Select a random pattern
    pattern_id = random.choice(list(patterns.keys()))
    pattern_label = patterns[pattern_id]["label"]
    pattern_probability = patterns[pattern_id]["probability"]
    pattern_max_drawdown = patterns[pattern_id]["max_drawdown"]

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

    return predicted_price, pattern_id, pattern_label, pattern_probability, pattern_max_drawdown, twitter_sentiment, news_sentiment, impact_score, news_summary

# CLI Simulation
def main():
    print("\n=== Stock Market Prediction CLI ===\n")
    
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
    predicted_price, pattern_id, pattern_label, pattern_probability, pattern_max_drawdown, twitter_sentiment, news_sentiment, impact_score, news_summary = get_stock_prediction(stock_name, prediction_date)

    # Display the results
    print("=== Prediction Results ===")
    print(f"Stock Selected: {stock_name}")
    print(f"Prediction Date: {prediction_date}")
    print(f"Predicted Next Price: ${predicted_price}")
    print(f"Pattern ID: {pattern_id}")
    print(f"Pattern Label: {pattern_label}")
    print(f"Pattern Probability: {pattern_probability}%")
    print(f"Pattern Maximum Drawdown (Risk): {pattern_max_drawdown}%")
    print(f"Predicted Twitter Sentiment Score: {twitter_sentiment}")
    print(f"Predicted News Sentiment Score: {news_sentiment}")
    print(f"Predicted Impact Score: {impact_score}")
    print(f"Summary of the News: {news_summary}")
    print("\nThank you for using the Stock Market Prediction CLI!")

if __name__ == "__main__":
    main()
