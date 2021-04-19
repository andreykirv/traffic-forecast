import streamlit as st
import pandas as pd
import plotly.express as px
from traffic_forecast import make_forecast

st.set_page_config(layout="wide")
scatter_column, settings_column = st.beta_columns((5, 2))
scatter_column.title("Traffic Forecast")
settings_column.header("Settings")
settings_column.write('Use .csv file with two columns: date, sessions.')
uploaded_file = settings_column.file_uploader("Choose File")
settings_column.subheader('Choose prediction horizon:')
period = settings_column.slider('', 30, 365, 365)
settings_column.subheader('Pick seasonalities:')
ys = settings_column.checkbox('Yearly seasonaity',True)
ws = settings_column.checkbox('Weekly seasonality')
settings_column.subheader('Set Uncertainty Interval Width:')
iw = settings_column.slider('The possible range of trend', 0.8, 1.0, 0.95, 0.05)
settings_column.subheader('Set changepoint scale:')
cps = settings_column.slider('It will make the trend more flexible', 0.05, 2.0, 0.05,0.1)


if uploaded_file is not None:
    data_import = pd.read_csv(uploaded_file)
    data_import.columns = ['ds', 'y']
    pred, trend = make_forecast(data_import, period, ys, ws, iw, cps)
    scatter_column.plotly_chart(pred, True)
    scatter_column.title('Trends')
    scatter_column.plotly_chart(trend, True)
else:
    scatter_column.header("Please Choose a file")
