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

def make_forecast(df, period = 365, ys = True, ws = False, iw = 0.95, cps=0.05):
    model = Prophet(weekly_seasonality=ws,
                    yearly_seasonality=ys,
                    interval_width=iw,
                    changepoint_prior_scale=cps,
                    )
    #model.add_country_holidays(country_name='UA')
    model.fit(df)
    future_dates = model.make_future_dataframe(periods=period)
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
    layout = dict(title=f'Traffic Forecast for {period} days',
                yaxis_title='GA Sessions',
                xaxis_title="Date",
                margin=dict(l=0, r=20, t=70, b=0),
                showlegend=True,
                legend=dict(
                    x=0,
                    y=0
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
    fig2.update_layout(margin=dict(l=0, r=20, t=70, b=0),
                    title='Trend of potentional growth',
                    yaxis_title='GA Sessions',
                    xaxis_title="Date"
                    )
    fig2.update_layout(margin=dict(l=0, r=20, t=70, b=0),
                    title='Trend of potentional growth',
                    yaxis_title='GA Sessions',
                    xaxis_title="Date"
                    )

    return fig, fig2
