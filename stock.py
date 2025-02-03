import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error

# Page Configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📊", layout="wide")

# Custom header
st.markdown("""
    <div style="background-color:#4CAF50;padding:15px;border-radius:10px;">
        <h1 style="color:white;text-align:center;">Stock Data Analyzer</h1>
    </div>
""", unsafe_allow_html=True)

# Footer with credits
st.markdown("""
    <style>
        footer {visibility: hidden;} /* Hide default Streamlit footer */
        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: black;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 18px;
            z-index: 9999;
        }
    </style>
    <div class="custom-footer">
        Developed by Pranav Maheshwari | Stock Data Project
    </div>
""", unsafe_allow_html=True)

# Header Section
st.markdown(
    '<div class="header animated-heading"><h1>Stock Data Analyzer</h1></div>',
    unsafe_allow_html=True,)

# Predefined stock symbols with full names and additional information
# Updated stock symbols with 20 entries
stock_symbols = {
    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "founder": "J. R. D. Tata",
        "history": "Founded in 1968, TCS is a part of the Tata Group and is India's largest IT services company.",
        "present_condition": "TCS remains a leading global IT services company, ranked among the top technology providers worldwide."
    },
    "INFY.NS": {
        "name": "Infosys Limited",
        "founder": "N. R. Narayana Murthy",
        "history": "Founded in 1981, Infosys pioneered the Indian IT outsourcing model and revolutionized software exports.",
        "present_condition": "Infosys is a top IT company in India with strong global presence in consulting and technology services."
    },
    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "founder": "Dhirubhai Ambani",
        "history": "Reliance started in 1966 as a textiles business and expanded into petrochemicals, telecom, and retail.",
        "present_condition": "Reliance is India's largest private sector company, leading in energy, telecom, and retail sectors."
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank",
        "founder": "Hasmukhbhai Parekh",
        "history": "HDFC Bank was incorporated in 1994 and became a key player in India's banking industry.",
        "present_condition": "HDFC Bank is one of the largest private sector banks in India, offering financial services to millions."
    },
    "ICICIBANK.NS": {
        "name": "ICICI Bank",
        "founder": "ICICI Group",
        "history": "Founded in 1994, ICICI Bank has been a leader in technological advancements in banking.",
        "present_condition": "ICICI Bank continues to dominate the banking sector, focusing on retail and corporate banking."
    },
    "SBIN.NS": {
        "name": "State Bank of India",
        "founder": "British India",
        "history": "Established in 1955, SBI evolved from the Imperial Bank of India and is the largest public sector bank in India.",
        "present_condition": "SBI is the backbone of India's banking system with a massive customer base and digital outreach."
    },
    "WIPRO.NS": {
        "name": "Wipro Limited",
        "founder": "M.H. Hasham Premji",
        "history": "Founded in 1945, Wipro began as a vegetable oil manufacturer before transitioning to IT services.",
        "present_condition": "Wipro is now one of India's top IT services firms with global presence."
    },
    "HCLTECH.NS": { 
        "name": "HCL Technologies",
        "founder": "Shiv Nadar",
        "history": "HCL was established in 1976 as a hardware company, later evolving into a global IT services firm.",
        "present_condition": "HCL continues to innovate in IT services, products, and consulting."
    },
    "BHARTIARTL.NS": {
        "name": "Bharti Airtel",
        "founder": "Sunil Bharti Mittal",
        "history": "Founded in 1995, Airtel is one of the leading telecom operators in India.",
        "present_condition": "Airtel provides broadband, mobile, and digital TV services globally."
    },
    "ITC.NS": {
        "name": "ITC Limited",
        "founder": "British American Tobacco Company",
        "history": "Established in 1910, ITC started as a tobacco company and later diversified.",
        "present_condition": "ITC operates in FMCG, hotels, paperboards, and agriculture."
    },
    "LT.NS": {
        "name": "Larsen & Toubro",
        "founder": "Henning Holck-Larsen and Søren Kristian Toubro",
        "history": "Founded in 1938, L&T is an Indian multinational engaged in EPC projects and high-tech manufacturing.",
        "present_condition": "L&T is a leader in engineering, construction, and financial services."
    },
    "ASIANPAINT.NS": {
        "name": "Asian Paints",
        "founder": "Champaklal H. Choksey and partners",
        "history": "Founded in 1942, Asian Paints grew to become India's largest and Asia's third-largest paint company.",
        "present_condition": "Asian Paints leads in decorative and industrial coatings."
    },
    "MARUTI.NS": {
        "name": "Maruti Suzuki",
        "founder": "Government of India (with Suzuki collaboration)",
        "history": "Founded in 1981, Maruti revolutionized Indian car manufacturing.",
        "present_condition": "Maruti is India's leading automobile manufacturer."
    },
    "TITAN.NS": {
        "name": "Titan Company",
        "founder": "Tata Group and TIDCO",
        "history": "Established in 1984, Titan started with watches and expanded to jewelry and eyewear.",
        "present_condition": "Titan dominates India's lifestyle product market."
    },
    "BAJAJFINSV.NS": {
        "name": "Bajaj Finserv",
        "founder": "Bajaj Group",
        "history": "Founded in 2007, Bajaj Finserv focuses on financial services and insurance.",
        "present_condition": "Bajaj Finserv is a leader in India's financial sector."
    },
    "ADANIGREEN.NS": {
        "name": "Adani Green Energy",
        "founder": "Adani Group",
        "history": "Established in 2015, Adani Green specializes in renewable energy projects.",
        "present_condition": "It is one of the largest renewable energy companies in India."
    },
    "COALINDIA.NS": {
        "name": "Coal India Limited",
        "founder": "Government of India",
        "history": "Founded in 1975, Coal India is the largest coal producer in the world.",
        "present_condition": "Coal India plays a pivotal role in India's energy production."
    },
    "ULTRACEMCO.NS": {
        "name": "UltraTech Cement",
        "founder": "Birla Group",
        "history": "Established in 1983, UltraTech is a leader in cement manufacturing.",
        "present_condition": "UltraTech is India's largest cement manufacturer."
    },
    "SUNPHARMA.NS": {
        "name": "Sun Pharmaceutical",
        "founder": "Dilip Shanghvi",
        "history": "Founded in 1983, Sun Pharma started with a focus on psychiatry products.",
        "present_condition": "Sun Pharma is now India's largest pharmaceutical company."
    },
    "BAJAJ-AUTO.NS": {
        "name": "Bajaj Auto",
        "founder": "Jamnalal Bajaj",
        "history": "Established in 1945, Bajaj Auto revolutionized two-wheeler manufacturing.",
        "present_condition": "Bajaj Auto is a leading manufacturer of motorcycles and scooters globally."
    }
}

# Sidebar inputs
st.sidebar.header("Enter Stock Details")

# Convert stock symbols dictionary to list of display names
stock_options = [f"{symbol} - {details['name']}" for symbol, details in stock_symbols.items()]
selected_stock_option = st.sidebar.selectbox("Select Stock Symbol", stock_options)

# Extract the stock symbol from the selected option
stock = selected_stock_option.split(" - ")[0]

# Calendar date range from 2000 to today
min_date = datetime(2000, 1, 1)
max_date = datetime.today()

start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", datetime(2025, 1, 1), min_value=min_date, max_value=max_date)

# Function to fetch stock data
def get_stock_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        return None
    return data

def calculate_stock_metrics(data):
    highest_price = data['Close'].max()
    lowest_price = data['Close'].min()

    # Get the index (timestamp) of the highest and lowest price
    best_time_to_sell = data['Close'].idxmax()  # Timestamp for max price
    best_time_to_buy = data['Close'].idxmin()   # Timestamp for min price

    # Ensure we are working with a Timestamp, not a Series
    if isinstance(best_time_to_sell, pd.Series):
        best_time_to_sell = best_time_to_sell.iloc[0]  # Extract the first element if it's a Series
    if isinstance(best_time_to_buy, pd.Series):
        best_time_to_buy = best_time_to_buy.iloc[0]  # Extract the first element if it's a Series

    # Convert to string format
    best_time_to_sell_str = best_time_to_sell.strftime('%Y-%m-%d')
    best_time_to_buy_str = best_time_to_buy.strftime('%Y-%m-%d')

    average_price = data['Close'].mean()

    metrics = {
        "Highest Price": highest_price,
        "Lowest Price": lowest_price,
        "Best Time to Sell": best_time_to_sell_str,
        "Best Time to Buy": best_time_to_buy_str,
        "Average Price": average_price
    }
    return metrics


if st.sidebar.button("Fetch Data"):
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        stock_info = stock_symbols[stock]
        st.markdown(f"""
            <h1 style="font-size: 36px;text-align:center;">{stock_info['name']}</h1>
            <p style="font-size: 22px;"><b>Founder:</b> {stock_info['founder']}</p>
            <p style="font-size: 22px;"><b>History:</b> {stock_info['history']}</p>
            <p style="font-size: 22px;"><b>Present Condition:</b> {stock_info['present_condition']}</p>
        """, unsafe_allow_html=True)

        # Add space between text and graph
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Fetch stock data
        data = get_stock_data(stock, start_date, end_date)
        
        if data is not None and 'Close' in data.columns:
            # Calculate stock metrics
            metrics = calculate_stock_metrics(data)

            # Extract scalar values from pandas Series
            highest_price = metrics['Highest Price'].iloc[0] if isinstance(metrics['Highest Price'], pd.Series) else metrics['Highest Price']
            lowest_price = metrics['Lowest Price'].iloc[0] if isinstance(metrics['Lowest Price'], pd.Series) else metrics['Lowest Price']
            average_price = metrics['Average Price'].iloc[0] if isinstance(metrics['Average Price'], pd.Series) else metrics['Average Price']

            # Use st.markdown with custom CSS for styling the text
            st.markdown(
                f"""
                <h3 style="font-size: 22px;">Highest Price: ₹{highest_price:.2f}</h3>
                <h3 style="font-size: 22px;">Lowest Price: ₹{lowest_price:.2f}</h3>
                 <h3 style="font-size: 22px;">Average Price: ₹{average_price:.2f}</h3>
                <h3 style="font-size: 22px;">Best Time to Sell: {metrics['Best Time to Sell']}</h3>
                <h3 style="font-size: 22px;">Best Time to Buy: {metrics['Best Time to Buy']}</h3>
                """, unsafe_allow_html=True
            )


        
        # Display stock data table
        st.markdown(
        f"<h2 style='text-align: center;'>Stock Data for {stock_symbols[stock]['name']}</h2>",
        unsafe_allow_html=True
        )
        st.write(data)

        # Plot Closing Price
        st.markdown("<h3 style='text-align: center; font-size: 36px;'>Stock Closing Price</h3>", unsafe_allow_html=True)
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label="Closing Price", color='blue')
        plt.title(f"{stock_symbols[stock]['name']} Stock Closing Price")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)


        # Add space between ARIMA Table and Plot Closing Price
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Forecasting with ARIMA Model
        st.markdown(
        "<h3 style='text-align: center; font-size: 36px;'>ARIMA Model Forecast</h3>",
        unsafe_allow_html=True
        )

        arima_model = ARIMA(data['Close'], order=(5, 1, 0))  # Simple ARIMA model (p=5, d=1, q=0)
        arima_model_fit = arima_model.fit()
        forecast = arima_model_fit.forecast(steps=30)

        # ARIMA Accuracy
        def arima_accuracy(data, forecast):
            actual_values = data['Close'][-30:].values
            mape = mean_absolute_percentage_error(actual_values, forecast)
            return 100 - (mape * 100)  # Return accuracy percentage

        arima_accuracy_value = arima_accuracy(data, forecast)
        st.markdown(f"<h3>ARIMA Model Accuracy: {arima_accuracy_value:.2f}%</h3>", unsafe_allow_html=True)

        # Plot ARIMA Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label="Actual Close Price", color='blue')
        plt.plot(pd.date_range(data.index[-1], periods=31, freq='B')[1:], forecast, label="Forecasted", color='red')
        plt.title(f"ARIMA Forecast for {stock_symbols[stock]['name']}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # ARIMA Forecast Table
        arima_forecast_dates = pd.date_range(data.index[-1], periods=31, freq='B')[1:]
        arima_forecast_df = pd.DataFrame({'Date': arima_forecast_dates, 'Forecasted Price': forecast})
        arima_forecast_df['Date'] = pd.to_datetime(arima_forecast_df['Date'])
        arima_forecast_df['Forecasted Price'] = arima_forecast_df['Forecasted Price'].round(2)
        
        st.markdown(
        "<br><br><h2 style='text-align: center;'>ARIMA Forecast Table</h2><br>",
        unsafe_allow_html=True)
        st.dataframe(arima_forecast_df)

        # Add space between ARIMA Table and LSTM Section
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Prepare data for LSTM
        st.markdown("<br><br><h3 style='text-align: center; font-size: 36px;'>LSTM Model Forecast</h3><br><br>", unsafe_allow_html=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Create LSTM Dataset
        def create_lstm_dataset(data, time_step=60):
            x_data, y_data = [], []
            for i in range(time_step, len(data)):
                x_data.append(data[i-time_step:i, 0])
                y_data.append(data[i, 0])
            return np.array(x_data), np.array(y_data)

        x_data, y_data = create_lstm_dataset(scaled_data)

        # Reshape for LSTM
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
        y_data = y_data.reshape(-1, 1)

        # Build LSTM Model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(x_data, y_data, epochs=10, batch_size=32)

        # Predict with LSTM
        lstm_predictions = lstm_model.predict(x_data)
        lstm_predictions = np.squeeze(lstm_predictions)
        lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))

        # LSTM Accuracy
        def lstm_accuracy(actual_values, predicted_values):
            mape = mean_absolute_percentage_error(actual_values, predicted_values)
            return 100 - (mape * 100)  # Return accuracy percentage

        actual_values_lstm = data['Close'][-len(lstm_predictions):].values
        lstm_accuracy_value = lstm_accuracy(actual_values_lstm, lstm_predictions.flatten())
        st.markdown(f"<h3>LSTM Model Accuracy: {lstm_accuracy_value:.2f}%</h3>", unsafe_allow_html=True)

        # Plot LSTM Predictions
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label="Actual Close Price", color='blue')
        plt.plot(data.index[-len(lstm_predictions):], lstm_predictions, label="LSTM Prediction", color='orange')
        plt.title(f"LSTM Model Prediction for {stock_symbols[stock]['name']}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # LSTM Prediction Table with rounded values
        st.markdown("<br><br><h2 style='text-align: center;'>LSTM Model Prediction Table</h2><br>", unsafe_allow_html=True)
        lstm_dates = data.index[-len(lstm_predictions):]
        lstm_df = pd.DataFrame({'Date': lstm_dates, 'Predicted Price': lstm_predictions.flatten()})
        lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
        lstm_df['Predicted Price'] = lstm_df['Predicted Price'].round(2)  # Round to 2 decimal places
        st.dataframe(lstm_df)
