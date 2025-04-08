import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Set up email details from environment variables
sender_email = "yoonusk2001@gmail.com"
password = "jwnp hink pciz slzq"
smtp_server = "smtp.gmail.com"
smtp_port = 587

# Define the receiver email
receiver_email = "it@arabiangates.com"

subject = "Welcome to Amazing Company"
body = """\
Hey there!

Thanks for joining us at Amazing Company. Your email has been verified and your account has been created.
Head to the website to login and start using our features."""

# Create the email message
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject

# Attach the email body to the message
message.attach(MIMEText(body, "plain"))

# Establish a connection to the SMTP server and send the email
try:
    # Connect to the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, password)  # Log in to the SMTP server
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )  # Send the email
    print("Email sent successfully")
except Exception as e:
    print(f"Error: {e}")