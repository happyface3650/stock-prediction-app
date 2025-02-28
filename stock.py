import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from prophet.forecaster import Prophet

Prophet._float_type = np.float64

#sets start and end time of the data fetched and graphs drawn
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")
#search stocks by their tickers
selected_stock = st.text_input("Enter Stock ticker")

#slider to control the number of days at which stock prices are predicted in the graph below
n_days = st.slider("Days predicted:", 1, 1000)
period = n_days

#loads data from yfinance for the selected stock
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
def plot_raw_data(data): #plots the data 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening Prices'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing Prices'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True) #slider to adjust the range of x axis
        st.plotly_chart(fig)
if len(selected_stock) != 0: #checks if characters are entered
    current_price_container = st.container()
    data_load_state = st.text("Load data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading data ...done!")
    if data is None:
        st.write("Please enter a valid stock ticker.")
    else:
        ticker = yf.Ticker(selected_stock) #gets current price of the stock from yfinance
        ticker_info = ticker.info
        
        if 'currentPrice' in ticker_info:#checks if the market is open
            current_price = ticker_info['currentPrice']
            st.subheader(f"Current Price: ${round(current_price,2)}")
        else: # if the market is closed
            value = data.iloc[-1]['Close']
            rounded_value = round(value, 2)
            st.subheader(f"Last closing price: ${rounded_value}")
        
        plot_raw_data(data)
        st.subheader('Raw data')
        table = data.tail() #table of highs, lows, opening and closing prices from the past few days
        table = table.drop(columns=['Adj Close'])
        tb = pd.DataFrame(table)
        tb_reset = tb.reset_index(drop=True)
        st.dataframe(tb_reset)
        # Forecasting
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
        if df_train.dropna().shape[0] >= 2: # checks if data is sufficient
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future) #predict future prices
            st.subheader('Forecast data')
            forecast_copy = forecast.copy() #create a copy of forecast data
            #select relevant coumns for display
            forecast_copy = forecast_copy.drop(columns=['trend_upper', 'trend', 'trend_lower', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper'])
            #rename columns 
            forecast_copy = forecast_copy.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Prices', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
            predicted_close = round(forecast_copy.iloc[-1]['Forecasted Prices'], 2)
            st.write(f"Price after {n_days} days: ${predicted_close}")
            fig1 = plot_plotly(m, forecast) #combine predicted and actual prices
            st.plotly_chart(fig1) #plot the graph
            st.subheader('Forecast components')
            fig2 = m.plot_components(forecast) #shows trend lines for different time of the day, week and year
            st.write(fig2)
            fc = pd.DataFrame(forecast_copy.tail())
            fc_reset = fc.reset_index(drop=True)#display data
            st.dataframe(fc_reset)
        else: #error message if the stock ticker is wrong 
            st.error("Invalid Stock Ticker/Not enough data to make a forecast. Please try again.") 
            
