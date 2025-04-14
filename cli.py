
import datetime
import engine_v2
engine = engine_v2.EnhancedPredictionEngine()
# Simulated company stock list
companies = {
    1: "GOLD (GLD)",
    2: "BTC (BTC)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

def get_stock_prediction(stock_id, date):
    
    engine.main_function(stock_id, date)
    print("\nThank you for using the Stock Market Prediction CLI!")
# CLI Simulation
def main():
    
    print("\n=== Stock Market Prediction CLI ===\n")
    
    # log in to the system , ask for username and password
    while True:
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        if engine.db.login(username, password):
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
            date_input = input("\nEnter the date for prediction (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS): ").strip()
            
            # Try parsing with time first
            try:
                prediction_date = datetime.datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # If that fails, try parsing just the date and add midnight time
                prediction_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
                prediction_date = prediction_date.replace(hour=0, minute=0, second=0)
            
            if prediction_date > datetime.datetime.now():
                print("Date must not be in the future.")
            else:
                formatted_date = prediction_date.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Selected date for prediction: {formatted_date}")
                print("\nFetching real-time stock and sentiment data...\n")
                get_stock_prediction(stock_id, formatted_date)
                break
                
        except ValueError:
            print("Invalid date format. Please enter either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")

# Run the main function

if __name__ == "__main__":
    main()
