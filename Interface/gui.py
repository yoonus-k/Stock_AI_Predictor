"""
Stock AI Predictor GUI Module

This module provides a Streamlit-based graphical user interface for the Stock AI Predictor.
It allows users to login, select stocks, make predictions, and view/export the results.
"""

###############################################################################
# IMPORTS
###############################################################################
# Standard library imports
import datetime
from io import BytesIO

# Third-party imports
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Local application imports
from Core.engine_v2 import EnhancedPredictionEngine
import Interface.send_email as send_email
from Data.Database.db import Database

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

def plot_price_prediction_chart(price_series, current_price, predicted_price, max_gain, max_drawdown, isPlot=True):
    """
    Plot a chart showing the price prediction and related markers.
    
    Args:
        price_series: Series of historical prices
        current_price: Current price value
        predicted_price: Predicted price value
        max_gain: Maximum expected gain 
        max_drawdown: Maximum expected drawdown
        isPlot: Whether to display the plot in Streamlit immediately
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot price line
    ax.plot(price_series.index, price_series.values, color="lightgray", linewidth=2)

    # Last time index for dot placement
    last_time = price_series.index[0]

    # Plot dots + text labels directly
    ax.plot(last_time, current_price, 'o', color='orange')
    ax.text(last_time, current_price, '  Current Price', color='orange', verticalalignment='bottom', fontsize=9)

    ax.plot(last_time, predicted_price, 'o', color='blue')
    ax.text(last_time, predicted_price, '  Predicted Price', color='blue', verticalalignment='bottom', fontsize=9)

    ax.plot(last_time, current_price * (1 + max_gain), 'o', color='green')
    ax.text(last_time, current_price * (1 + max_gain), '  Max Gain Target', color='green', verticalalignment='bottom', fontsize=9)

    ax.plot(last_time, current_price * (1 + max_drawdown), 'o', color='red')
    ax.text(last_time, current_price * (1 + max_drawdown), '  Max Drawdown Limit', color='red', verticalalignment='bottom', fontsize=9)

    ax.set_title("📉 Recent Price Movement with Prediction Points")
    ax.set_ylabel("Price")
    ax.grid(True)

    if isPlot:
        st.pyplot(fig)
    
    return fig

def download_fancy_pdf(report_text, chart_fig, prediction_data):
    """
    Generate a PDF report containing prediction data and charts.
    
    Args:
        report_text: Text content for the report
        chart_fig: Matplotlib figure with the prediction chart
        prediction_data: Dictionary containing prediction metrics
        
    Returns:
        BytesIO buffer containing the PDF data
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("📊 Stock AI Prediction Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Summary Text
    elements.append(Paragraph("Prediction Summary", styles['Heading2']))
    for line in report_text.split("\n"):
        elements.append(Paragraph(line, styles['Normal']))

    elements.append(Spacer(1, 12))

    # Fancy Metrics Table
    pattern = prediction_data['pattern_metrics']
    data = [
        ["Metric", "Value"],
        ["Predicted Price", f"${prediction_data['final_prediction']:.2f}"],
        ["Max Gain", f"{pattern['max_gain']*100:.2f}%"],
        ["Max Drawdown", f"{pattern['max_drawdown']*100:.2f}%"],
        ["Reward/Risk Ratio", f"{pattern['reward_risk_ratio']:.2f}"],
        ["Confidence", f"{prediction_data['confidence']*100:.1f}%"],
        ["Action", prediction_data['action']]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
    ]))
    elements.append(table)

    elements.append(Spacer(1, 12))

    # Convert matplotlib chart to image and add to PDF
    img_buffer = BytesIO()
    chart_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=250))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def logout():
    """
    Logout the user by clearing the session state and closing database connections.
    """
    # Close the database connection if it exists
    if "db_instance" in st.session_state and st.session_state.db_instance:
        st.session_state.db_instance.close()
    # Clear the session
    st.session_state.clear()

###############################################################################
# MAIN APPLICATION
###############################################################################
def main():
    """
    Main application function with Streamlit interface.
    """
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="Stock AI Predictor", layout="wide")
    st.title("📊 Stock AI Predictor GUI")
    st.markdown("An AI-powered platform for predicting stock market movements using pattern recognition and sentiment analysis.")

    # --- INITIALIZE SESSION STATE ---
    initialize_session_state()
    
    # --- DATABASE AND ENGINE INITIALIZATION ---
    if "db_instance" not in st.session_state:
        st.session_state.db_instance = Database()
        
    # Initialize engine
    engine = EnhancedPredictionEngine(db=st.session_state.db_instance)
    
    # --- LOGIN SCREEN ---
    if not st.session_state.logged_in:
        display_login_screen(engine)
        st.stop()

    # --- SIDEBAR ---
    display_sidebar()
    
    # --- PREDICTION FORM ---
    handle_prediction_form(engine)
    
    # --- DISPLAY RESULTS ---
    if st.session_state.prediction_data:
        display_prediction_results()
    
    # --- FOOTER ---
    display_footer()

def initialize_session_state():
    """Initialize all required session state variables if they don't exist."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None
    if "prediction_id" not in st.session_state:
        st.session_state.prediction_id = None
    if "prediction_report" not in st.session_state:
        st.session_state.prediction_report = ""
    if "CurrentPrices" not in st.session_state:
        st.session_state.CurrentPrices = None
    if "Price_Plot" not in st.session_state:
        st.session_state.Price_Plot = None

def display_login_screen(engine):
    """Display the login form and handle authentication."""
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if st.session_state.db_instance.login(username, password):
            st.session_state.logged_in = True
            st.session_state.user_email = engine.db.get_user_email(username)
            st.session_state.user_id = engine.db.get_user_id(username)
            st.success("✅ Login successful!")
            st.rerun()  # force refresh to show sidebar after login
        else:
            st.error("❌ Invalid credentials.")

def display_sidebar():
    """Display the main sidebar options after login."""
    st.sidebar.header("📊 Prediction Settings")
    
    # --- STOCK SELECTION ---
    stock_id = st.sidebar.selectbox("Select Stock", list(COMPANIES.keys()), format_func=lambda x: COMPANIES[x])
    date_input = st.sidebar.date_input("Select Date", value=datetime.date.today())
    time_input = st.sidebar.time_input("Select Time", value=datetime.time(hour=16, minute=0))
    
    # Store in session state for access in other functions
    st.session_state.stock_id = stock_id
    st.session_state.full_datetime = datetime.datetime.combine(date_input, time_input)

def handle_prediction_form(engine):
    """Handle the prediction form submission and generate results."""
    if st.sidebar.button("🔍 Predict"):
        # Check if the date is in the future
        if st.session_state.full_datetime > datetime.datetime.now():
            st.error("❌ Prediction is not available for future dates.")
            st.stop()
        
        # Create a placeholder message
        try:
            status_msg = st.empty()
            status_msg.info("⏳ Running prediction engine...")
            report, data = engine.main_function(
                st.session_state.stock_id, 
                st.session_state.full_datetime.strftime("%Y-%m-%d %H:%M:%S")
            )
            data = convert_numpy_types(data)
        except Exception as e:
            st.error(f"❌ Failed to run prediction engine: {str(e)}")
            st.stop()
        
        # Get stock data and plot prediction chart
        df = engine.get_stock_data(
            st.session_state.stock_id, 
            st.session_state.full_datetime.strftime("%Y-%m-%d %H:%M:%S"), 
            7
        )
        recent = df[df.index <= st.session_state.full_datetime].head(24)  # Last 24 records
        
        # Save current price to session state
        st.session_state.CurrentPrices = recent['ClosePrice']
        st.session_state.Price_Plot = plot_price_prediction_chart(
            st.session_state.CurrentPrices,
            current_price=data["current_price"],
            predicted_price=data["final_prediction"],
            max_gain=data["pattern_metrics"]["max_gain"],
            max_drawdown=data["pattern_metrics"]["max_drawdown"]
        )

        # Save prediction to session
        st.session_state.prediction_data = data
        st.session_state.prediction_report = report

        # Store in DB
        prediction_id = st.session_state.db_instance.store_prediction_data(st.session_state.stock_id, data)
        st.session_state.prediction_id = prediction_id
        
        # Replace the status message with success
        status_msg.success("✅ Prediction complete. Ready to send email or download report.")

def display_prediction_results():
    """Display the prediction results in a nicely formatted way."""
    # Display metrics in a row
    col1, col2, col3 = st.columns(3)
    col1.metric("Action", st.session_state.prediction_data["action"])
    col2.metric("Confidence", f"{st.session_state.prediction_data['confidence']*100:.1f}%")
    col3.metric("Max Risk", f"{st.session_state.prediction_data['pattern_metrics']['max_drawdown']*100:.2f}%")
    
    # Display pattern metrics summary
    st.markdown("### 📊 Pattern Metrics Summary")
    pattern = st.session_state.prediction_data["pattern_metrics"]
    st.table({
        "Type": [pattern["type"]],
        "Probability": [f"{pattern['probability']:.2%}"],
        "Max Gain": [f"{pattern['max_gain']:.2%}"],
        "Max Drawdown": [f"{pattern['max_drawdown']:.2%}"],
        "Reward/Risk Ratio": [f"{pattern['reward_risk_ratio']:.2f}"]
    })
  
    # Display sentiment analysis summary
    st.markdown("### 🧠 Sentiment Analysis")
    sentiment = st.session_state.prediction_data["sentiment_metrics"]
    twitter_sentiment = st.session_state.prediction_data["twitter_sentiment"]
    with st.expander("News Summary", True):
        st.write(f"📰 **News Sentiment Score**: `{sentiment['Predicted News Sentiment Score']:.2f}`")
        st.write(f"🐦 **Twitter Score**: `{twitter_sentiment['tweets_weighted_sentiment_score']:.2f}`")
        st.write(f"📊 **Impact Score**: `{sentiment['Predicted Impact Score']:.2f}`")
        st.write(f"📈 **Summary of News**: '{sentiment['Summary of the News']}`")

    # Display all Chart, Report, and Sentiment in tabs
    tab1, tab2, tab3 = st.tabs(["📉 Price Chart", "📄 Report", "🧠 Sentiment"])
    with tab1:
        st.pyplot(st.session_state.Price_Plot)

    with tab2:
        st.code(st.session_state.prediction_report)

    with tab3:
        st.write(sentiment)

    # --- Export Options ---
    display_export_options()

def display_export_options():
    """Display options to export or send the prediction report."""
    # Send Email Button
    if st.button("📤 Send Report via Email"):
        current_time = datetime.datetime.now()
        send_email.send_email(st.session_state.prediction_data, st.session_state.user_email)
        st.session_state.db_instance.store_notification_data(
            st.session_state.user_id, 
            st.session_state.prediction_id, 
            current_time, 
            'Email', 
            'Sent'
        )
        st.success("📧 Email sent successfully to " + st.session_state.user_email)

    # Download as PDF Button
    if st.button("📄 Download Report as PDF"):
        fig = plot_price_prediction_chart(
            st.session_state.CurrentPrices,
            current_price=st.session_state.CurrentPrices.iloc[-1],
            predicted_price=st.session_state.prediction_data["final_prediction"],
            max_gain=st.session_state.prediction_data["pattern_metrics"]["max_gain"],
            max_drawdown=st.session_state.prediction_data["pattern_metrics"]["max_drawdown"],
            isPlot=False  # Don't plot it again, we will use the saved figure
        )
        
        buffer = download_fancy_pdf(
            st.session_state.prediction_report, 
            fig, 
            st.session_state.prediction_data
        )

        st.download_button(
            label="📥 Click here to download PDF",
            data=buffer,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )

def display_footer():
    """Display the footer with logout option."""
    st.sidebar.markdown("---")
    st.sidebar.write("Logged in as:", st.session_state.user_email)
    # Button to logout
    st.sidebar.button("🚪 Logout", on_click=logout)

###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()

