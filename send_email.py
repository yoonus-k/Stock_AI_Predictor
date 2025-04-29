from email.message import EmailMessage
import smtplib
import ssl
import os
from dotenv import load_dotenv
load_dotenv()



email_sender = "stock.ai.predictor@gmail.com"
email_password = os.getenv("EMAIL_KEY")
email_receiver = "it@arabiangates.com"

def send_email(prediction_data, email_receiver):
        # Email content
    subject = "Stock Market Prediction Report"
    body = prediction_data
    # parse the prediction data into a string and format it for email
    body = f"Prediction Data:\n\n"
    for key, value in prediction_data.items():
        if isinstance(value, dict):
            body += f"{key}:\n"
            for sub_key, sub_value in value.items():
                body += f"  {sub_key}: {sub_value}\n"
        else:
            body += f"{key}: {value}\n"
            
            
    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(body)


    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
        print("Email sent successfully")