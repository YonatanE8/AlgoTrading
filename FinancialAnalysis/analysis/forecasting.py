from abc import ABC
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from FinancialAnalysis.analysis.smoothing import Smoother

import numpy as np


class Forecaster(ABC):
    """
    Class for managing all forecasting methods for a 1D time series
    """

    def __init__(
            self,
            method: str = 'smoother',
            forecast_horizon: int = 5,
            smoother: Smoother = None,
            arima_orders: (int, int, int) = (1, 0, 1),
            arima_prediction_type: str = 'levels',
            sarimax_orders: (int, int, int) = (1, 0, 1),
            sarimax_seasonal_order: (int, int, int, int) = (0, 0, 0, 0),
            remove_mean: bool = False
    ):
        """
        Initialize a Forecaster class for producing future predictions of a time-series.

        :param method: (str) Method to use for forecasting, can be one of:
        'smoother': Use the exponential / polynomial parameters of a fitted Smoother
        for forecasting future values.
        'arima': Use an Autoregressive Integrated Moving Average model.
        'sarimax: Use Seasonal AutoRegressive Integrated Moving Average with
        exogenous regressors model.
        :param forecast_horizon: (int) Number of future temporal-points to predict.
        Default is 5 (i.e. a trading week).
        :param smoother: (Smoother) Used with the 'smoother' mehtod. A fitted Smoother
        class, based on which to perform future forecasts. Default to None.
        :param arima_orders: ((int, int, int)) Used with the 'arima' method.
        The orders for the ARIMA model. Should be specified as (p, d, q) where
        'p' is the auto-regressive model order, 'd' is the integrative model order,
        and 'q' is the moving-average model order. Default is (1, 0, 1).
        :param sarimax_orders: (int, int, int) The (p,d,q) order of the model for the
        number of AR parameters, differences, and MA parameters.
        d must be an integer indicating the integration order of the process,
        while p and q may either be an integers indicating the AR and MA orders.
        Default is (1,0,1).
        :param sarimax_seasonal_order: (int, int, int, int) The (P,D,Q,s) orders of the
        seasonal component of the model for the AR parameters, differences,
        MA parameters, and periodicity. D must be an integer indicating the
        integration order of the process, while P and Q may either be an integers
        indicating the AR and MA orders (so that all lags up to
        those orders are included) or else iterables giving specific AR and / or MA
        lags to include. s is an integer giving the periodicity
        (number of periods in season),
        often it is 4 for quarterly data or 12 for monthly data.
        Default is no seasonal effect, i.e. (0, 0, 0, 0).
        :param arima_prediction_type: (str) Used with the 'arima' method. Determines
        the methodology of producing predictions from the ARIMA model. Can be either
        ‘linear’ : Linear prediction in terms of the differenced endogenous variables.
        ‘levels’ : Predict the levels of the original endogenous variables.
        Default is 'levels'. For further details please refer to:
        https://www.statsmodels.org -> statsmodels.tsa.arima_model.ARIMAResults.predict.
        :param remove_mean: (bool) Whether to normalize the data by removing
        the mean, might be useful when using the ARIMA model, which should operate on a
        stationary process.
        """

        # Check inputs
        assert method in ('smoother', 'arima', 'sarimax'), \
            f"The {method} forecasting method is not currently supported. Please " \
            f"use one of the following: 'smoother', 'arima'."
        assert not (method == 'smoother' and smoother is None), \
            f"Cannot use the 'smoother' forecasting method without supplying a " \
            f"fitted instance of the Smoother class."

        # Set parameters
        self._method = method
        self._forecast_horizon = forecast_horizon
        self._smoother = smoother
        self._arima_orders = arima_orders
        self._sarimax_orders = sarimax_orders
        self._sarimax_seasonal_order = sarimax_seasonal_order
        self._arima_prediction_type = arima_prediction_type
        self._remove_mean = remove_mean

        # Setup
        self._arima_model = None
        self._sarimax_model = None
        self._mean = 0

    def _smooth_forecast_exp(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method for producing an exponential-based predictions from the Smoother
        class.

        :param time_series: (NumPy array) The time-series on which to compute forecasts

        :return: (NumPy array) The future forecast
        """

        if self._smoother.exp_smoother is None:
            self._smoother(time_series=time_series)

        forecast = self._smoother.exp_smoother.predict(
            start=len(time_series), end=(len(time_series) + self._forecast_horizon - 1))

        return forecast

    def _smooth_forecast_poly(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method for producing a polynomial-based predictions from the Smoother
        class.

        :param time_series: (NumPy array) The time-series on which to compute forecasts

        :return: (NumPy array) The future forecast
        """

        if self._smoother.poly is None:
            self._smoother(time_series=time_series)

        x = np.arange(len(time_series), (len(time_series) + self._forecast_horizon))
        forecast = self._smoother.poly(x)

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

        if self._smoother.method in ('exp', 'holt_winter'):
            forecast = self._smooth_forecast_exp(time_series=time_series)

        elif self._smoother.method == 'avg':
            forecast = [np.mean(time_series)]
            for i in range(1, min(self._forecast_horizon, (len(time_series) - 1))):
                series = np.concatenate(
                    [time_series[i:], np.array(forecast)]
                )
                forecast.append(np.mean(series))

            forecast = np.array(forecast)

        elif self._smoother.method == 'polyfit':
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

        if self._arima_model is None:
            self._arima_model = ARIMA(time_series, order=self._arima_orders).fit()

        if prediction_start is None or prediction_end is None:
            prediction_start = len(time_series)
            prediction_end = len(time_series) + self._forecast_horizon - 1

        forecast = self._arima_model.predict(start=prediction_start, end=prediction_end,
                                             typ=self._arima_prediction_type)

        return forecast

    def _sarimax_forecast(self, time_series: np.ndarray, prediction_start: int,
                          prediction_end: int) -> np.ndarray:
        """
        Utility method, fits & uses an SARIMAX model based on the parameters given
        at construction for producing future returns.

        :param time_series: (NumPy array) The time-series on which to compute forecasts
        :param prediction_start: (int) Index from which to start predictions,
        in relation to the total array used for fitting.
        :param prediction_end: (int) Index by which to end predictions,
        in relation to the total array used for fitting.

        :return: (NumPy array) The future forecast
        """

        if self._sarimax_model is None:
            self._sarimax_model = SARIMAX(
                time_series,
                order=self._sarimax_orders,
                seasonal_order=self._sarimax_seasonal_order,
            ).fit()

        if prediction_start is None or prediction_end is None:
            prediction_start = len(time_series)
            prediction_end = len(time_series) + self._forecast_horizon - 1

        forecast = self._sarimax_model.predict(
            start=prediction_start,
            end=prediction_end,
        )

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

        if self._remove_mean:
            self._mean = np.mean(time_series, axis=-1)
            time_series -= self._mean

        if self._method == 'smoother':
            forecast = self._smooth_forecast(time_series=time_series) + self._mean

        elif self._method == 'arima':
            forecast = self._arima_forecast(
                time_series=time_series,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            ) + self._mean

        elif self._method == 'sarimax':
            forecast = self._sarimax_forecast(
                time_series=time_series,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            ) + self._mean

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

        if self._method == 'smoother':
            description = f"Smoothing based forecast: {self._smoother.description}"

        if self._method == 'arima':
            description = f"ARIMA based forecast: Orders = {self._arima_orders}"

        return description

    @property
    def forecast_horizon(self) -> int:
        """
        :return: Returns the forecaster future forecast horizon
        """

        return self._forecast_horizon

    def reset(self) -> None:
        """
        Utility method for resetting the any fitted model parameters, useful for when
        performing predictions based on a rolling window.

        :return: None
        """

        if self._smoother is not None:
            if self._smoother.method in ('avg', 'exp', 'holt_winter'):
                self._smoother.exp_smoother = None

            elif self._smoother.method == 'polyfit':
                self._smoother.poly = None

        self._arima_model = None
        self._sarimax_model = None
