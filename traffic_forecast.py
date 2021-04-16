import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_absolute_error
import numpy as np
from numpy import array
from fbprophet.plot import plot_plotly

def make_forecast(df):
    model = Prophet(weekly_seasonality=False)
    model.fit(df)
    future_dates = model.make_future_dataframe(periods=365)
    prediction = model.predict(future_dates)


    fig = plot_plotly(model, prediction)
    fig.update_layout(
        title='Traffic Forecast for 365 days', yaxis_title='GA Sessions', xaxis_title="Date"
    )

    return fig
