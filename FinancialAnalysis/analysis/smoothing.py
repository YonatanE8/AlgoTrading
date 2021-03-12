from abc import ABC
from scipy.signal import convolve
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

import numpy as np


class Smoother(ABC):
    """
    Class for managing all smoothing methods for a 1D time series
    """

    def __init__(self, method: str = 'avg', length: int = 5, alpha: float = 0.6,
                 optimize: bool = False, trend: str = None, poly_degree: int = 2):
        """
        Constructor for the Smoother class, determines the smoothing method and its
        parameters

        :param method: (str) Method to use for smoothing, can be one of:
        'avg': Averaging over a running window,
        'exp': Exponential smoothing,
        'holt_winter': Exponential smoothing using the Holt-Winter method,
        'polyfit': Polynomial fitting,
        :param length: (int) Used with the 'avg' method. The length of the
         averaging window in days. Default is 5 (i.e. trading week).
        :param alpha: (float) Used with the 'exp' method. The smoothing factor,
        only used if 'optimize' is False. Default is 0.6.
        :param optimize: (bool) Used with the 'exp' method. Determines whether to
        compute the optimal alpha by minimizing the SSE function from the observations.
        Default is False
        :param trend: (str) Used with the 'holt_winter' method.
        The trend type the data presents, one of:
        "add", "mul", "additive", "multiplicative", None.
        Default to None.
        :param poly_degree: (str) Used with the '_fit_polyfit' method.
        The degree of the fitted polynomial. Default is 2.
        """

        # Check inputs
        assert method in ('avg', 'exp', 'holt_winter', 'polyfit'), \
            f"The {method} smoothing method is not currently supported. Please " \
            f"use one of the following: 'avg', 'exp', 'holt_winter'."

        # Set parameters
        self.method = method
        self.length = length
        self.alpha = alpha
        self.optimize = optimize
        self.trend = trend
        self.poly_degree = poly_degree

        # Initialize placeholder for future use by forecasters
        self.poly = None
        self.exp_smoother = None

    def _running_window_smoothing(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method which takes in a NumPy array representing a time-series,
        and output the a smoothed time series, by using a running-window averaging with
        a window length 'length'. Note that this method returns only the 'same'
        segment of the convolution, hence the outputted signal size will be identical
        to the inputted one.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        # Define the averaging window
        window = (1 / self.length) * np.ones((self.length,))
        smoothed_time_series = convolve(time_series, window, mode='valid',
                                        method='auto')

        return smoothed_time_series

    def _exponential_smoothing(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method which takes in a NumPy array representing a time-series,
        and output the exponentially smoothed time series.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        if self.optimize:
            exp_smooth = SimpleExpSmoothing(time_series).fit(optimized=True)
            self.alpha = exp_smooth.params['smoothing_level']

        else:
            exp_smooth = SimpleExpSmoothing(time_series).fit(smoothing_level=self.alpha,
                                                             optimized=False)

        self.exp_smoother = exp_smooth
        smoothed_time_series = exp_smooth.fittedvalues

        return smoothed_time_series

    def _holt_winters_smoothing(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method which takes in a NumPy array representing a time-series,
        and output the Holt-Winter smoothed time series.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        exp_smooth = ExponentialSmoothing(time_series, damped_trend=False,
                                          trend=self.trend).fit()
        self.exp_smoother = exp_smooth
        smoothed_time_series = exp_smooth.fittedvalues

        return smoothed_time_series

    def _fit_polyfit(self, time_series: np.ndarray) -> np.ndarray:
        """
        Utility method which takes in a NumPy array representing a time-series,
        fit a polynomial of order 'poly_degree' given at construction,
        and returns the polynomial over the same x-axis range.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        x = np.arange(len(time_series))
        poly = Polynomial.fit(x, time_series, deg=self.poly_degree)
        self.poly = poly
        fitted_time_series = poly(x)

        return fitted_time_series

    def smooth(self, time_series: np.ndarray) -> np.ndarray:
        """
        Wrapper method around the various smoothing methods implemented in the Smoother
        class. Takes in a NumPy array representing a time-series to smooth, and returns
        the smoothed series based on the specified parameters in the constructor.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        if self.method == 'avg':
            smoothed = self._running_window_smoothing(time_series=time_series)

        elif self.method == 'exp':
            smoothed = self._exponential_smoothing(time_series=time_series)

        elif self.method == 'holt_winter':
            smoothed = self._holt_winters_smoothing(time_series=time_series)

        elif self.method == 'polyfit':
            smoothed = self._fit_polyfit(time_series=time_series)

        return smoothed

    def __call__(self, time_series: np.ndarray) -> np.ndarray:
        """
        Implementation of the 'Call' functionality of Smoother, which calls the 'smooth'
        method.

        :param time_series: (NumPy array) The time-series to smooth

        :return: (NumPy array) The smoothed time-series
        """

        return self.smooth(time_series=time_series)

    @property
    def description(self) -> str:
        """
        Smoother instance Property. Generates a string which describes to current
        instance in terms of method and parameters used for smoothing.

        :return: (str) Description for the current instance of the Smoother class
        """

        if self.method == 'avg':
            description = f"Average Window: Length = {self.length}"

        if self.method == 'exp':
            description = f"Exponential Smoothing: alpha = {self.alpha}"

        if self.method == 'holt_winter':
            description = f"Holt-Winter Smoothing: trend = {self.trend}"

        if self.method == 'polyfit':
            description = f"PolyFit: Order = {self.poly_degree}"

        return description


