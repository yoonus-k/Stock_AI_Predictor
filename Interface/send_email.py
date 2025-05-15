"""
Stock AI Predictor Email Module

This module provides email functionality for the Stock AI Predictor.
It handles sending prediction reports via email to users.
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import ssl
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

###############################################################################
# CONSTANTS
###############################################################################
# Email credentials
EMAIL_SENDER = "stock.ai.predictor@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_KEY")
DEFAULT_RECEIVER = "it@arabiangates.com"

###############################################################################
# EMAIL FUNCTIONS
###############################################################################
def send_email(prediction_data, email_receiver=DEFAULT_RECEIVER):
    """
    Send prediction data via email to the specified recipient.
    
    Args:
        prediction_data (dict): Prediction data to include in the email
        email_receiver (str): Email address of the recipient
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Email content
    subject = "Stock Market Prediction Report"
    
    # Parse the prediction data into a string and format it for email
    body = f"Prediction Data:\n\n"
    for key, value in prediction_data.items():
        if isinstance(value, dict):
            body += f"{key}:\n"
            for sub_key, sub_value in value.items():
                body += f"  {sub_key}: {sub_value}\n"
        else:
            body += f"{key}: {value}\n"
    
    # Create email message
    em = EmailMessage()
    em["From"] = EMAIL_SENDER
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(body)

    # Setup secure connection and send email
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, email_receiver, em.as_string())
            print("Email sent successfully")
            return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False