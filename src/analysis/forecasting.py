from abc import ABC
from src.analysis.smoothing import Smoother
from statsmodels.tsa.arima_model import ARIMA

import numpy as np


class Forecaster(ABC):
    """
    Class for managing all forecasting methods for a 1D time series
    """

    def __init__(self, method: str = 'smoother', forecast_horizon: int = 5,
                 smoother: Smoother = None,
                 arima_orders: (int, int, int) = (10, 3, 10),
                 arima_prediction_type: str = 'levels'):
        """
        Initialize a Forecaster class for producing future predictions of a time-series.

        :param method: (str) Method to use for forecasting, can be one of:
        'smoother': Use the exponential / polynomial parameters of a fitted Smoother
        for forecasting future values.
        'arima': Use an Autoregressive Integrated Moving Average model.
        :param forecast_horizon: (int) Number of future temporal-points to predict.
        Default is 5 (i.e. a trading week).
        :param smoother: (Smoother) Used with the 'smoother' mehtod. A fitted Smoother
        class, based on which to perform future forecasts. Default to None.
        :param arima_orders: ((int, int, int)) Used with the 'arima' method.
        The orders for the ARIMA model. Should be specified as (p, d, q) where
        'p' is the auto-regressive model order, 'd' is the integrative model order,
        and 'q' is the moving-average model order. Default is (10, 3, 10).
        :param arima_prediction_type: (str) Used with the 'arima' method. Determines
        the methodology of producing predictions from the ARIMA model. Can be either
        ‘linear’ : Linear prediction in terms of the differenced endogenous variables.
        ‘levels’ : Predict the levels of the original endogenous variables.
        Default is 'levels'. For further details please refer to:
        https://www.statsmodels.org -> statsmodels.tsa.arima_model.ARIMAResults.predict
        """

        # Check inputs
        assert method in ('smoother', 'arima'), \
            f"The {method} forecasting method is not currently supported. Please " \
            f"use one of the following: 'smoother', 'arima'."
        assert not (method == 'smoother' and smoother is None), \
            f"Cannot use the 'smoother' forecasting method without supplying a " \
            f"fitted instance of the Smoother class."

        # Set parameters
        self.method = method
        self.forecast_horizon = forecast_horizon
        self.smoother = smoother
        self.arima_orders = arima_orders
        self.arima_prediction_type = arima_prediction_type

        # Setup
        self.arima_model = None

    def _smooth_forecast_exp(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method for producing an exponential-based predictions from the Smoother
        class.

        :param time_series: (NumPy array) The time-series on which to compute forecasts

        :return: (NumPy array) The future forecast
        """

        if self.smoother.exp_smoother is None:
            self.smoother(time_series=time_series)

        forecast = self.smoother.exp_smoother.predict(
            start=len(time_series), end=(len(time_series) + self.forecast_horizon - 1))

        return forecast

    def _smooth_forecast_poly(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method for producing a polynomial-based predictions from the Smoother
        class.

        :param time_series: (NumPy array) The time-series on which to compute forecasts

        :return: (NumPy array) The future forecast
        """

        if self.smoother.poly is None:
            self.smoother(time_series=time_series)

        x = np.arange(len(time_series),
                      (len(time_series) + self.forecast_horizon))
        forecast = self.smoother.poly(x)

        return forecast

    def _smooth_forecast(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method, uses the Smoother instance given at construction for producing
        future returns. Uses the Smoother 'method' param to decide on which parameters
        to use when making a prediction. If method is 'avg' then uses the 'exp'
        parameters. If the parameters are not yet fitted, fits them on the fly.

        :param time_series: (NumPy array) The time-series on which to compute forecasts

        :return: (NumPy array) The future forecast
        """

        if self.smoother.method in ('avg', 'exp', 'holt_winter'):
            forecast = self._smooth_forecast_exp(time_series=time_series)

        elif self.smoother.method == 'polyfit':
            forecast = self._smooth_forecast_poly(time_series=time_series)

        return forecast

    def _arima_forecast(self, time_series: np.ndarray, prediction_start: int,
                        prediction_end: int) -> np.ndarray:
        """
        Utility method, fits & uses an ARIMA model based on the parameters given
        at construction for producing future returns.

        :param time_series: (NumPy array) The time-series on which to compute forecasts
        :param prediction_start: (int) Index from which to start predictions,
        in relation to the total array used for fitting.
        :param prediction_end: (int) Index by which to end predictions,
        in relation to the total array used for fitting.

        :return: (NumPy array) The future forecast
        """

        if self.arima_model is None:
            self.arima_model = ARIMA(time_series, order=self.arima_orders).fit()

        if prediction_start is None or prediction_end is None:
            prediction_start = len(time_series)
            prediction_end = len(time_series) + self.forecast_horizon - 1

        forecast = self.arima_model.predict(start=prediction_start, end=prediction_end,
                                            typ=self.arima_prediction_type)

        return forecast

    def forecast(self, time_series: np.ndarray, prediction_start: int = None,
                 prediction_end: int = None) -> np.ndarray:
        """
        Wrapper method around the various forecasting methods implemented in the
        Forecaster class. Takes in a NumPy array representing a time-series for which
        forecasts are required, and returns the forecast.

        :param time_series: (NumPy array) The time-series on which to compute forecasts
        :param prediction_start: (int) Used only with ARIMA predictions.
        Index from which to start predictions, in relation to the total array
        used for fitting. Defaults is None.
        :param prediction_end: (int) Used only with ARIMA predictions.
        Index by which to end predictions, in relation to the total array
        used for fitting. Defaults is None.

        :return: (NumPy array) The future forecast
        """

        if self.method == 'smoother':
            forecast = self._smooth_forecast(time_series=time_series)

        elif self.method == 'arima':
            forecast = self._arima_forecast(time_series=time_series,
                                            prediction_start=prediction_start,
                                            prediction_end=prediction_end)

        return forecast

    def __call__(self, time_series: np.ndarray, prediction_start: int = None,
                 prediction_end: int = None) -> np.ndarray:
        """
        Implementation of the 'Call' functionality of Forecaster,
        which calls the 'forecast' method.

        :param time_series: (NumPy array) The time-series on which to compute forecasts
        :param prediction_start: (int) Used only with ARIMA predictions.
        Index from which to start predictions, in relation to the total array
        used for fitting. Defaults is None.
        :param prediction_end: (int) Used only with ARIMA predictions.
        Index by which to end predictions, in relation to the total array
        used for fitting. Defaults is None.

        :return: (NumPy array) The future forecast
        """

        return self.forecast(time_series=time_series, prediction_start=prediction_start,
                             prediction_end=prediction_end)

    @property
    def description(self) -> str:
        """
        Forecaster instance Property. Generates a string which describes to current
        instance in terms of method and parameters used for smoothing.

        :return: (str) Description for the current instance of the Smoother class
        """

        if self.method == 'smoother':
            description = f"Smoothing based forecast: {self.smoother.description}"

        if self.method == 'arima':
            description = f"ARIMA based forecast: Orders = {self.arima_orders}"

        return description








