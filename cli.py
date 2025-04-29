
import datetime
import engine_v2
import send_email
import numpy as np

engine = engine_v2.EnhancedPredictionEngine()
# Simulated company stock list
companies = {
    1: "GOLD (GLD)",
    2: "BTC (BTC)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}
user_email = None
user_id = None
# Convert numpy types to native Python types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj
def get_stock_prediction(stock_id, date):
    # get the stock prediction and sentiment data
    prediction_report , prediction = engine.main_function(stock_id, date)
    prediction = convert_numpy_types(prediction)
   
    # store the prediction data in the database
    prediction_id =engine.db.store_prediction_data(stock_id, prediction)
    
    # send the prediction data to the user email
    send_email.send_email(prediction, user_email)
    
    # store the notification data in the database
    engine.db.store_notification_data(user_id, prediction_id, date,'Email', 'Sent')
    engine.db.close()
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
            global user_email
            global user_id
            user_email = engine.db.get_user_email(username)
            user_id = engine.db.get_user_id(username)
            
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
            stock_choice = int(input("\nEnter stock number (0 to exit) : "))
            # exit if user inputs 0
            if stock_choice == 0:
                print("Exiting the program.")
                engine.db.close()
                return
            if stock_choice in companies:
                stock_name = companies[stock_choice]
                stock_id = stock_choice
                
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
                        # if the date is after 2025-04-11, then the prediction is not available
                        elif prediction_date > datetime.datetime(2025, 4, 12):
                            print("Prediction is not available for future dates after 2025-04-11.")
                        else:
                            formatted_date = prediction_date.strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Selected date for prediction: {formatted_date}")
                            print("\nFetching real-time stock and sentiment data...\n")
                            get_stock_prediction(stock_id, formatted_date)
                            break
                            
                    except ValueError:
                        print("Invalid date format. Please enter either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")
                
            else:
                print("Invalid choice. Please select a valid stock number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    

# Run the main function

if __name__ == "__main__":
    main()
