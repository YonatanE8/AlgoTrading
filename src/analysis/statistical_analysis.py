from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

import numpy as np


def exponential_smoothing(time_series: np.ndarray, alpha: float = 0.6,
                          optimize: bool = False) -> np.ndarray:
    """
    This method takes in a NumPy array representing a time-series, and output the
    exponentially smoothed time series, of the same shape,
    according to the specified parameters.

    :param time_series: (NumPy array) The time-series to smooth
    :param alpha: (float) smoothing factor, only used if 'optimize' is False.
    Default is 0.6.
    :param optimize: (bool) Whether to compute the optimal alpha by minimizing the SSE
    function from the observations. Default is False

    :return: (NumPy array) The smoothed time-series
    """

    if optimize:
        exp_smooth = SimpleExpSmoothing(time_series).fit(optimized=True)

    else:
        exp_smooth = SimpleExpSmoothing(time_series).fit(smoothing_level=alpha,
                                                         optimized=False)

    smoothed_time_series = exp_smooth.fittedvalues

    return smoothed_time_series


def holt_winters_smoothing(time_series: np.ndarray, trend: str = None) -> np.ndarray:
    """
    This method takes in a NumPy array representing a time-series, and output the
    exponentially smoothed time series, of the same shape,
    based on the HoltWinters method.

    :param time_series: (NumPy array) The time-series to smooth
    :param trend: (str) The trend type the data presents, one of:
    "add", "mul", "additive", "multiplicative", None. Default to None.

    :return: (NumPy array) The smoothed time-series
    """

    exp_smooth = ExponentialSmoothing(time_series, damped=False, trend=trend).fit()
    smoothed_time_series = exp_smooth.fittedvalues

    return smoothed_time_series



