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

st.title("Stock Prediction App")
        #search stocks by their tickers
selected_stock = st.text_input("Enter Stock ticker")
        

        #slider to control the number of days at which stock prices are predicted in the graph below
n_days = st.slider("Days predicted:", 1, 1000)
period = n_days
if selected_stock: #checks if characters are entered
        stock = selected_stock.upper()
        print(stock)
        close_column = f'Close_{stock}'

        print(close_column)
        open_column = f'Open_{stock}'
        print(open_column)
        current_price_container = st.container()
        data_load_state = st.text("Load data...")
        data = load_data(selected_stock)
        print(data.head())
        if data.empty:
                st.error("Invalid stock ticker or Yfinance exceeded rate limit.")
        else:
                if close_column not in data.columns:
                    st.error(f"The {close_column} column is missing in the data.")
                else:
                    try:
                        ticker = yf.Ticker(selected_stock)
                        if not ticker.info:
                            st.error("Invalid stock ticker. Please try again.")
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
                    data_load_state.text("Loading data ...done!")
                    ticker_info = ticker.info
                    current_price = ticker_info['currentPrice']
                    if current_price is not None: #checks if the market is open
                        st.subheader(f"Current price:{current_price}")
                    else: # if the market is closed
                        value = data.iloc[-1][close_column]
                        rounded_value = round(value, 2)
                        st.subheader(f"Current price:{rounded_value}")

                    st.subheader('Raw data') #table of highs, lows, opening and closing prices from the past few days
                    st.write(data.tail())
                    print(data.columns)
                    plot_raw_data(data, open_column, close_column)
                    # Forecasting
                    data[close_column] = pd.to_numeric(data[close_column], errors='coerce')
                    data = data.dropna(subset=[close_column])
                    data['Date_'] = pd.to_datetime(data['Date_'])
                    df_train = data[['Date_', close_column]].rename(columns={"Date_": "ds", close_column: "y"})
                    print(df_train.tail())

                    
                    if df_train.dropna().shape[0] >= 2: # checks if data is sufficient
                        m = Prophet()
                        m.fit(df_train)
                        future = m.make_future_dataframe(periods=period)
                        forecast = m.predict(future) #predict future prices
                        st.subheader('Forecast data')
                        forecast_copy = forecast.copy() #create a copy of forecast data
                        #select relevant coumns for display
                        forecast_copy = forecast_copy.drop(columns=['trend_upper', 'trend_lower', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper'])
                        #rename columns 
                        forecast_copy = forecast_copy.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Prices', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
                        st.write(forecast_copy.tail()) #display data
                        fig1 = plot_plotly(m, forecast) #combine predicted and actual prices
                        st.plotly_chart(fig1) #plot the graph
                        st.subheader('Forecast components')
                        fig2 = m.plot_components(forecast) #shows trend lines for different time of the day, week and year
                        st.write(fig2)
                else: #error message if the stock ticker is wrong 
                        st.error("Invalid Stock Ticker/Not enough data to make a forecast. Please try again.") 
