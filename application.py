import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from traffic_forecast import make_forecast, validate

def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv" target="_blank">Download Forecast in .csv</a>'
    return href

st.set_page_config(layout="wide")
scatter_column, settings_column = st.beta_columns((5, 2))

scatter_column.title("Traffic Forecast")


settings_column.header("Settings")
settings_column.write('Use .csv file with two columns: date, sessions.')
uploaded_file = settings_column.file_uploader("Choose File")
data_type = settings_column.radio('Choose data type to forecast:',['Daily', 'Monthly'])
if data_type == 'Daily':
    settings_column.subheader('Choose prediction horizon:')
    period = settings_column.slider('', 30, 365, 365)
if data_type == 'Monthly':
    settings_column.subheader('Choose prediction horizon:')
    period = settings_column.slider('', 1, 12, 12)
settings_column.subheader('Pick seasonalities:')
ys = settings_column.checkbox('Yearly seasonaity',False)
ws = settings_column.checkbox('Weekly seasonality')
settings_column.subheader('Set Uncertainty Interval Width:')
iw = settings_column.slider('The possible range of trend', 0.8, 1.0, 0.95, 0.05)
settings_column.subheader('Set changepoint scale:')
cps = settings_column.slider('It will make the trend more flexible', 0.05, 2.0, 0.05,0.1)


if uploaded_file is not None:
    data_import = pd.read_csv(uploaded_file)
    data_import.columns = ['ds', 'y']
    pred, trend, df = make_forecast(data_import, period, ys, ws, iw, cps, data_type)
    scatter_column.plotly_chart(pred, True)
    scatter_column.title('Trends')
    scatter_column.plotly_chart(trend, True)
    settings_column.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
    validate_bool = settings_column.button('Validate the Forecast', help='Train the model on 80% of the original dataset and compare the forecast of the remaining 20% ​​with real data.')
    if validate_bool:
        chart, mae = validate(data_import, ys, ws, iw, cps, data_type)
        scatter_column.plotly_chart(chart,True)
        scatter_column.write(f'Mean Absolute Error is equal to {round(mae,1)}.')
else:
    scatter_column.header("Please Choose a file")
