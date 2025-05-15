"""
Stock AI Predictor CLI Module

This module provides a command-line interface for the Stock AI Predictor.
It allows users to login, select stocks, make predictions, and receive reports via email.
"""

###############################################################################
# IMPORTS
###############################################################################
import datetime
import numpy as np

# Local application imports
import Core.engine_v2 as engine_v2
import Interface.send_email as send_email

###############################################################################
# CONSTANTS
###############################################################################
COMPANIES = {
    1: "GOLD (GLD)",
    2: "BTC (BTC)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: The object containing numpy types to convert
        
    Returns:
        Object with numpy types converted to native Python types
    """
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

def get_stock_prediction(stock_id, date, engine, user_email, user_id):
    """
    Get stock prediction, store it in the database, and send via email.
    
    Args:
        stock_id (int): ID of the stock to predict
        date (str): Date for the prediction
        engine: Prediction engine instance
        user_email (str): Email address to send the report to
        user_id (int): ID of the user
        
    Returns:
        None
    """
    # Get the stock prediction and sentiment data
    prediction_report, prediction = engine.main_function(stock_id, date)
    prediction = convert_numpy_types(prediction)
   
    # Store the prediction data in the database
    prediction_id = engine.db.store_prediction_data(stock_id, prediction)
    
    # Send the prediction data to the user email
    send_email.send_email(prediction, user_email)
    
    # Current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store the notification data in the database
    engine.db.store_notification_data(user_id, prediction_id, current_date, 'Email', 'Sent')
    
    # Print report to console
    print("\n===== PREDICTION REPORT =====")
    print(prediction_report)
    print("============================\n")
    
    # Cleanup
    engine.db.close()
    print("\nThank you for using the Stock Market Prediction CLI!")

###############################################################################
# USER INTERFACE FUNCTIONS
###############################################################################
def handle_login(engine):
    """
    Handle user login process.
    
    Args:
        engine: Prediction engine instance
        
    Returns:
        tuple: (user_email, user_id) if login successful, or None if failed
    """
    print("\n=== Login ===")
    while True:
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        
        if engine.db.login(username, password):
            print("Login successful!")
            user_email = engine.db.get_user_email(username)
            user_id = engine.db.get_user_id(username)
            return user_email, user_id
        else:
            print("Invalid username or password.")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                return None, None

def handle_stock_selection():
    """
    Handle stock selection process.
    
    Returns:
        int: Selected stock ID or 0 to exit
    """
    print("\n=== Available Stocks ===")
    for key, value in COMPANIES.items():
        print(f"{key}. {value}")
    
    while True:
        try:
            stock_choice = int(input("\nEnter stock number (0 to exit): "))
            if stock_choice == 0 or stock_choice in COMPANIES:
                return stock_choice
            else:
                print("Invalid choice. Please select a valid stock number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def handle_date_selection():
    """
    Handle date selection process.
    
    Returns:
        str: Formatted date string or None if canceled
    """
    print("\n=== Date Selection ===")
    while True:
        try:
            date_input = input("Enter the date for prediction (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) or 'back' to return: ")
            
            if date_input.lower() == 'back':
                return None
                
            # Try parsing with time first
            try:
                prediction_date = datetime.datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # If that fails, try parsing just the date and add midnight time
                prediction_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
                prediction_date = prediction_date.replace(hour=0, minute=0, second=0)
            
            # Validate the date
            if prediction_date > datetime.datetime.now():
                print("Date must not be in the future.")
            # If the date is after 2025-04-11, then the prediction is not available
            elif prediction_date > datetime.datetime(2025, 4, 12):
                print("Prediction is not available for future dates after 2025-04-11.")
            else:
                formatted_date = prediction_date.strftime("%Y-%m-%d %H:%M:%S")
                return formatted_date
                
        except ValueError:
            print("Invalid date format. Please enter either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")

###############################################################################
# MAIN APPLICATION
###############################################################################
def main():
    """Main function to run the CLI interface."""
    print("\n========================================")
    print("=== Stock Market Prediction CLI v1.0 ===")
    print("========================================\n")
    
    # Initialize the prediction engine
    engine = engine_v2.EnhancedPredictionEngine()
    
    # Handle login
    user_email, user_id = handle_login(engine)
    if not user_email:
        print("Exiting the program.")
        engine.db.close()
        return
    
    # Main application loop
    while True:
        # Get stock selection
        stock_id = handle_stock_selection()
        if stock_id == 0:
            print("Exiting the program.")
            engine.db.close()
            return
        
        stock_name = COMPANIES[stock_id]
        print(f"\nSelected stock: {stock_name}")
        
        # Get date selection
        formatted_date = handle_date_selection()
        if not formatted_date:
            continue
            
        print(f"Selected date for prediction: {formatted_date}")
        print("\nFetching real-time stock and sentiment data...\n")
        
        # Get prediction
        get_stock_prediction(stock_id, formatted_date, engine, user_email, user_id)
        
        # Ask if user wants to make another prediction
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            print("Exiting the program.")
            engine.db.close()
            return

###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
