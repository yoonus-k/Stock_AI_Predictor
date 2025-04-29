import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from io import BytesIO
from engine_v2 import EnhancedPredictionEngine
import send_email
from io import BytesIO
if "db_instance" not in st.session_state:
    from Data.db import Database
    st.session_state.db_instance = Database()
# Initialize engine
engine = EnhancedPredictionEngine(db=st.session_state.db_instance)

# function to download report as PDF
def download_fancy_pdf(report_text, chart_fig, prediction_data):
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

# function to plot price prediction chart
def plot_price_prediction_chart(price_series, current_price, predicted_price, max_gain, max_drawdown, isPlot=True):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot price line
    ax.plot(price_series.index, price_series.values, color="lightgray", linewidth=2)

    # Last time index for dot placement
    last_time = price_series.index[-1]

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
    
# function to convert numpy types to native python types
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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock AI Predictor", layout="wide")
st.title("📊 Stock AI Predictor GUI")
st.markdown("An AI-powered platform for predicting stock market movements using pattern recognition and sentiment analysis.")

# --- SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "prediction_report" not in st.session_state:
    st.session_state.prediction_report = ""
if "CurrentPrices" not in st.session_state:
    st.session_state.CurrentPrices = None
    
# --- LOGIN ---
if not st.session_state.logged_in:
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
    st.stop()

# --- MAIN SIDEBAR ---
st.sidebar.header("📊 Prediction Settings")
companies = {
    1: "GOLD (GLD)",
    2: "BTC (BTC)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

# --- STOCK SELECTION ---
stock_id = st.sidebar.selectbox("Select Stock", list(companies.keys()), format_func=lambda x: companies[x])
date_input = st.sidebar.date_input("Select Date", value="2025-04-10")
time_input = st.sidebar.time_input("Select Time", value=datetime.time(hour=0, minute=0))
full_datetime = datetime.datetime.combine(date_input, time_input)

# --- PREDICT BUTTON ---
if st.sidebar.button("🔍 Predict"):
    # if the date is after 2025-04-11, show an error message
    if full_datetime > datetime.datetime(2025, 4, 12):
        st.error("❌ Prediction is not available for future dates after 2025-04-11.")
        st.stop()
    
    # Create a placeholder message
    status_msg = st.empty()
    status_msg.info("⏳ Running prediction engine...")
    report, data = engine.main_function(stock_id, full_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    data = convert_numpy_types(data)
    
    # plot price prediction chart
    df = st.session_state.db_instance.get_stock_data(stock_id)
    recent = df[df.index < full_datetime].tail(12)  # Last 48 records
    # save current price to session state
    st.session_state.CurrentPrices = recent['ClosePrice']
    plot_price_prediction_chart(
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
    prediction_id = st.session_state.db_instance.store_prediction_data(stock_id, data)
    st.session_state.db_instance.store_notification_data(
        st.session_state.user_id, prediction_id, full_datetime, 'Email', 'Pending'
    )
    
    # Replace the status message with success
    status_msg.success("✅ Prediction complete. Ready to send email or download report.")
    
   

# --- DISPLAY RESULTS IF PREDICTED ---
if st.session_state.prediction_data:
    # st.subheader("🧠 Final Prediction Report")
    # st.code(st.session_state.prediction_report)

    # Display metrics
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
    twitter_sentiment =  st.session_state.prediction_data["twitter_sentiment"]
    with st.expander("News Summary",True):
        st.write(f"📰 **News Sentiment Score**: `{sentiment['Predicted News Sentiment Score']:.2f}`")
        st.write(f"🐦 **Twitter Score**: `{twitter_sentiment['tweets_weighted_sentiment_score']:.2f}`")
        st.write(f"📊 **Impact Score**: `{sentiment['Predicted Impact Score']:.2f}`")
        st.write(f"📈 **Summary of News**: '{sentiment['Summary of the News']}`")


    # Display all Chart, Report, and Sentiment in tabs
    tab1, tab2, tab3 = st.tabs([ "📉 Price Chart","📄 Report", "🧠 Sentiment"])
    with tab1:
        plot_price_prediction_chart(
        st.session_state.CurrentPrices,
        current_price=st.session_state.CurrentPrices.iloc[-1],
        predicted_price=st.session_state.prediction_data["final_prediction"],
        max_gain=st.session_state.prediction_data["pattern_metrics"]["max_gain"],
        max_drawdown=st.session_state.prediction_data["pattern_metrics"]["max_drawdown"]
    )     
    with tab2:
        st.code(st.session_state.prediction_report)

    with tab3:
        st.write(sentiment)


    # --- Send Email Button ---
    if st.button("📤 Send Report via Email"):
        send_email.send_email(st.session_state.prediction_data, st.session_state.user_email)
        st.success("📧 Email sent successfully to " + st.session_state.user_email)

    # --- Download as PDF ---
    if st.button("📄 Download Report as PDF"):
        fig = plot_price_prediction_chart(
        st.session_state.CurrentPrices,
        current_price=st.session_state.CurrentPrices.iloc[-1],
        predicted_price=st.session_state.prediction_data["final_prediction"],
        max_gain=st.session_state.prediction_data["pattern_metrics"]["max_gain"],
        max_drawdown=st.session_state.prediction_data["pattern_metrics"]["max_drawdown"],
        isPlot=False  # Don't plot it again, we will use the saved figure
        )    # Use the chart generation function we made earlier
        
        buffer = download_fancy_pdf(st.session_state.prediction_report, fig, st.session_state.prediction_data)

        st.download_button(
            label="📥 Click here to download PDF",
            data=buffer,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Logged in as:", st.session_state.user_email)
def logout():
    # Close the database connection if it exists
    if "db_instance" in st.session_state and st.session_state.db_instance:
        st.session_state.db_instance.close()
    # Clear the session
    st.session_state.clear()

# Button to logout
st.sidebar.button("🚪 Logout", on_click=logout)

