import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by Shailendra Singh")
@st.cache_data
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df.copy() 

def main():
    st.title('Any Stock Price Predictions')

    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    
    bb_indicator = BollingerBands(data['Close'])
    bb = data.copy()  
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]
    
    
    macd = MACD(data['Close']).macd()
    rsi = RSIIndicator(data['Close']).rsi()
    sma = SMAIndicator(data['Close'], window=14).sma_indicator()
    ema = EMAIndicator(data['Close']).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data['Close'])
    elif option == 'BB':
        st.write('Bollinger Bands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Index')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    st.header('Stock Price Prediction')
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num_days = st.number_input('How many days forecast?', value=5, min_value=1)
    if st.button('Predict'):
        model_engine(model, num_days)

def model_engine(model_name, num_days):
    
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num_days)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num_days:]
    x = x[:-num_days]
    y = df['preds'].values
    y = y[:-num_days]

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)

    
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif model_name == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor()
    elif model_name == 'KNeighborsRegressor':
        model = KNeighborsRegressor()
    else:
        model = XGBRegressor()

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'R2 Score: {r2_score(y_test, preds)}\nMAE: {mean_absolute_error(y_test, preds)}')

    
    forecast_preds = model.predict(x_forecast)
    forecast_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, num_days + 1)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_preds})
    st.subheader('Forecasted Prices:')
    st.dataframe(forecast_df.set_index('Date'))
    
    
    st.subheader('Forecasted Prices Plot')
    st.line_chart(forecast_df.set_index('Date'))

if __name__ == '__main__':
    option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
    option = option.upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration (days)', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End Date', today)
    if st.sidebar.button('Download Data'):
        if start_date < end_date:
            st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
            data = download_data(option, start_date, end_date)
        else:
            st.sidebar.error('Error: End date must be after start date')
    
    
    scaler = StandardScaler()
    data = download_data(option, start_date, end_date)
    main()
