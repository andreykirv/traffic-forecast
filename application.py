import streamlit as st
import pandas as pd
import plotly.express as px
from traffic_forecast import make_forecast

st.set_page_config(layout="wide")
scatter_column, settings_column = st.beta_columns((5, 2))

scatter_column.title("Traffic Forecast")

settings_column.title("Settings")
settings_column.write('Use .csv file with two columns: date, sessions.')
uploaded_file = settings_column.file_uploader("Choose File")

if uploaded_file is not None:
    data_import = pd.read_csv(uploaded_file)
    data_import.columns = ['ds', 'y']
    pred = make_forecast(data_import)
    scatter_column.plotly_chart(pred)
else:
    scatter_column.header("Please Choose a file")
