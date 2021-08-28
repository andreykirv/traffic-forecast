import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_absolute_error
import numpy as np
from numpy import array
from fbprophet.plot import plot_plotly
from fbprophet.plot import plot_components_plotly
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
import streamlit as st
import holidays

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def make_forecast(df, period = 365, ys = True, ws = False, iw = 0.95, cps=0.05, data_type='Daily'):
    # Вкидываем праздники, для их учёта моделькой
    holidays_dict = holidays.UA(years=(2015, 2016, 2017, 2018, 2019, 2020, 2021))
    df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index') \
        .reset_index()
    df_holidays = df_holidays.rename({'index':'ds', 0:'holiday'}, axis ='columns')
    df_holidays['ds'] = pd.to_datetime(df_holidays.ds)
    df_holidays = df_holidays.sort_values(by=['ds'])
    df_holidays = df_holidays.reset_index(drop=True)

    model = Prophet(weekly_seasonality=ws,
                    yearly_seasonality=ys,
                    interval_width=iw,
                    changepoint_prior_scale=cps,
                    holidays=df_holidays
                    )

    model.fit(df)
    if data_type == 'Daily':
        future_dates = model.make_future_dataframe(periods=period)
    if data_type == 'Monthly':
        future_dates = model.make_future_dataframe(periods=period, freq='MS')
    prediction = model.predict(future_dates)

    actual_sessions = go.Scatter(
        name = 'Actual Sessions',
        mode = 'markers',
        fill=None,
        x = list(prediction['ds']),
        type='scatter',
        y = list(df['y']),
        marker=dict(
            color='#FFBAD2',
            line=dict(width=1))
    )
    trend = go.Scatter(
        name = 'Trend',
        mode = 'lines',
        x = list(prediction['ds']),
        y = list(prediction['yhat']),
        type='scatter',
        marker=dict(
            color='#eb3434',
            line=dict(width=3)
        )
    )
    upper_bound = go.Scatter(
        name = 'Upper Bound',
        mode = 'lines',
        x = list(prediction['ds']),
        type='scatter',
        y = list(prediction['yhat_upper']),
        line= dict(color='#57b8ff')
    )
    lower_bound = go.Scatter(
        name = 'Lower Bound',
        mode = 'lines',
        x = list(prediction['ds']),
        type='scatter',
        y = list(prediction['yhat_lower']),
        line= dict(color='#57b8ff'),
        fill='tonexty'
    )
    data = [upper_bound, lower_bound, actual_sessions, trend]
    if data_type == 'Daily':
        title = f'Traffic Forecast for {period} days'
    if data_type == 'Monthly':
        title = f'Traffic Forecast for {period} months'
    layout = dict(title=title,
                yaxis_title='GA Sessions',
                xaxis_title="Date",
                margin=dict(r=20, t=70, b=0),
                showlegend=True,
                legend=dict(
                    orientation='h',
                    x=0,
                    y=1.1
                ),
                xaxis=dict(
                    ticklen=5,
                    zerolinewidth=1
                ),
                yaxis=dict(
                    ticklen=5,
                    zerolinewidth=1
                )
            )

    fig=dict(data=data,layout=layout)

    fig2 = plot_components_plotly(model, prediction, figsize=(900,300))
    fig2.update_layout(margin=dict(r=20, t=70, b=0),
                    title='Trend of potentional growth',
                    yaxis_title='GA Sessions',
                    xaxis_title="Date"
                    )
    fig2.update_layout(margin=dict(l=0, r=20, t=70, b=0),
                    title='Trend of potentional growth',
                    yaxis_title='GA Sessions',
                    xaxis_title="Date"
                    )

    return fig, fig2, prediction[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def validate(df, ys = True, ws = False, iw = 0.95, cps=0.05, data_type='Daily'):
    dd = df.copy()
    predictions = int(round(dd.shape[0] * 0.8,0))-dd.shape[0]
    train = dd.drop(dd.index[predictions:])
    future_init = df.drop(df.index[:int(round(df.shape[0] * 0.8,0))])
    future = pd.DataFrame(future_init).reset_index(drop=True)

    if data_type == 'Daily':
        if future.shape[0] < 365:
            ws = True

    #We train the model
    model = Prophet(weekly_seasonality=ws,
                    yearly_seasonality=ys,
                    interval_width=iw,
                    changepoint_prior_scale=cps,
                    )
    model.fit(train)

    # We make the forecast
    forecast = model.predict(future)

    # We calculate the MAE between the actual values and the predicted values
    # Смотрим на ошибки модели
    cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))
    cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
    cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']
    mape = np.mean(abs(cmp_df[-predictions:]['p']))
    mae = np.mean(abs(cmp_df[-predictions:]['e']))

    # We plot the final output for a visual understanding
    y_true = future_init['y'].values
    y_pred = forecast['yhat'].values
    y_true = np.stack(y_true).astype(float)

    actual = go.Scatter(
        name = 'Actual',
        mode = 'lines',
        x = list(future['ds']),
        type='scatter',
        y = y_true,
        line= dict(color='#eb3434')
    )
    predicted = go.Scatter(
        name = 'Predicted',
        mode = 'lines',
        x = list(future['ds']),
        type='scatter',
        y = y_pred,
        line= dict(color='#57b8ff'),
        #fill='tonexty'
    )
    data = [actual, predicted]
    layout = dict(title=f'Comparing actual data with predicted',
                yaxis_title='GA Sessions',
                xaxis_title="Date",
                margin=dict(r=20, t=70, b=0),
                showlegend=True,
                legend=dict(
                    orientation='h',
                    x=0,
                    y=1.1
                ),
                xaxis=dict(
                    ticklen=5,
                    zerolinewidth=1
                ),
                yaxis=dict(
                    ticklen=5,
                    zerolinewidth=1
                )
            )

    fig=dict(data=data,layout=layout)

    return fig, mae, mape
