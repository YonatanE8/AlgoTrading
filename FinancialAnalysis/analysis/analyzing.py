from abc import ABC
from typing import Tuple

from scipy.signal import welch
from scipy.stats import linregress
from scipy.interpolate import interp1d
from FinancialAnalysis.stocks_io.data_queries import get_asset_data, get_multiple_assets

import numpy as np


class Analyzer(ABC):
    """
    Class for performing time-series based analysis over an asset's quote.
    """

    def __init__(self, symbols_list: tuple, start_date: str, end_date: str = None,
                 quote_channel: str = 'Close', adjust_prices: bool = True,
                 risk_free_asset_symbol: str = '^IRX', bins: int = 10,
                 spectral_energy_threshold: float = 0.001,
                 trend_period_length: int = 22, cache_path: str = None):
        """
        Constructor method for the Analyzer class

        :param symbols_list: (Tuple) A tuple with all of the symbols of the stocks
        for which data is to be queried.
        :param start_date: (str) Starting date, should be formatted as
        'year-month-day'".
        :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
        If None uses today's date. Defaults to None.
        :param quote_channel: (str) Specify the quote channel to perform analysis by.
         The available channels are: 'Close', 'Open', 'Low', 'High', 'Volume'.
         Default is 'Close'.
        :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
        defaults to True.
        :param risk_free_asset_symbol: (str) Which asset to refer to as the Risk-Free
        asset (for example for Sharpe-Ratio computation). Defaults to '^IRX',
        which refers to the yield of US treasury bonds for 13 weeks.
        :param bins: (int) Number of bins to use in the histogram analysis computation.
        Defaults to 10.
        :param spectral_energy_threshold: (float) threshold between 0 and 1 to use
        when performing the periodicty analysis. Defaults to 0.2.
        :param trend_period_length: (int) Period length (in trading days) to consider
        when performing trend analysis. Defaults to 22 (i.e. 1 trading month).
        :param cache_path: (str) Path to the directory in which to cache / look for
        cached data, if None does not use caching. Default is None.
        """

        # Setup
        self.start_date = start_date
        self.end_date = end_date
        self.quote_channel = quote_channel
        self.adjust_prices = adjust_prices
        self.risk_free_asset_symbol = risk_free_asset_symbol
        self.bins = bins
        self.spectral_energy_threshold = spectral_energy_threshold
        self.trend_period_length = trend_period_length

        # Query assets
        quotes, _, valid_symbols = get_multiple_assets(
            symbols_list=symbols_list,
            start_date=start_date,
            end_date=end_date,
            quote_channels=(quote_channel,),
            adjust_prices=adjust_prices,
            cache_path=cache_path,
        )
        quotes = quotes[quote_channel]

        # Interpolate over any NaNs
        x_axis = np.arange(quotes.shape[0])
        interpolated_quotes = [
            self._interpoloate(
                x_axis=x_axis,
                y=quotes[:, i],
                mode='previous',
            )
            for i in range(quotes.shape[1])
        ]
        interpolated_quotes = np.concatenate(
            [np.expand_dims(q, 0) for q in interpolated_quotes],
            0
        )

        self.n_assets = len(valid_symbols)
        self.symbols_list = valid_symbols
        self.quotes = interpolated_quotes.T
        self.returns = self._compute_returns(self.quotes)
        self.cumulative_returns = (self.returns + 1).prod(axis=0)

        # Query the risk free asset
        risk_free_asset, _ = get_asset_data(symbol=risk_free_asset_symbol,
                                            start_date=start_date, end_date=end_date,
                                            quote_channels=(quote_channel,),
                                            adjust_prices=adjust_prices,
                                            cache_path=cache_path)
        self.risk_free_asset = risk_free_asset[quote_channel]

        # Risk-free asset is already reported as returns, remove the first entry so that
        # the risk-free returns will align with the assets' returns.
        # We divide by 100 since the risk-free return is specified in %,
        # and divide by 255 in order to take into account the daily risk-free returns
        self.risk_free_returns = self.risk_free_asset[1:] / (100 * 255)

        # Place-holders
        self.spectral_components = None
        self.spectral_components_freqs = None
        self.spectral_components_energies = None

    def _compute_returns(self, quotes: np.ndarray) -> np.ndarray:
        """
        Utility method for computing returns

        :param quotes: (np.ndarray) The quotes for which to compute returns.

        :return: (np.ndarray) Returns of all assets
        """

        returns = np.diff(quotes, axis=0) / quotes[:-1]

        return returns

    @staticmethod
    def _interpoloate(
            x_axis: np.ndarray,
            y: np.ndarray,
            mode: str = 'previous') -> np.ndarray:
        """
        A utility method for performing interpolation, especially useful for computing
        the Sharpe-Ratio when the # of quotes for the risk-free asset is not complete.
        This method is wrapper around SciPy's interp1d method.

        :param x_axis: (np.ndarray) The x-axis to interpolate over
        :param y: (np.ndarray) The signal to interpolate
        :param mode: (str) The interpolation method.

        :return: (np.ndarray) The interpolated signal
        """

        # Start by removing the NaN values from 'y'
        not_nan_inds = ~np.isnan(y)
        y_without_nans = y[not_nan_inds]

        if not_nan_inds.shape[0] < x_axis.shape[0]:
            not_nan_inds = np.append(
                not_nan_inds,
                np.array([False] * (x_axis.shape[0] - not_nan_inds.shape[0])),
                0
            )

        x = x_axis[not_nan_inds]
        interpolator = interp1d(
            x,
            y_without_nans,
            kind=mode,
            fill_value="extrapolate",
        )
        interpolated_signal = interpolator(x_axis)

        return interpolated_signal

    def _analyze_sr(self) -> np.ndarray:
        """
        Utility method for computing the assets Sharpe-Ratio

        :return: (np.ndarray) Sharpe-Ratio of all specified assets over the period
        specified in the constructor.
        """

        # Compute the excess returns and standard-deviation of the excess returns
        if len(self.risk_free_returns) != self.returns.shape[0]:
            risk_free_returns = self._interpoloate(
                x_axis=np.arange(self.returns.shape[0]),
                y=self.risk_free_returns,
            )

        else:
            risk_free_returns = self.risk_free_returns

        excess_returns = self.returns - np.expand_dims(risk_free_returns, 1)
        excess_returns = np.cumprod((excess_returns + 1), 0)

        # Compute SR
        sr_risk = np.std(excess_returns, axis=0)
        sr_return = excess_returns[-1, :]
        sr = sr_return / sr_risk

        return sr

    @property
    def sr(self) -> np.ndarray:
        """
        Class property, the Sharpe-Ratio of the analyzed assets

        :return: (np.ndarray) An array with shape (T - 2, # assets) containing the
        Sharpe-Ratio of the analyzed assets. Where T is the temporal length of the
        requested assets.
        """

        return self._analyze_sr()

    def _analyze_returns_histogram(self) -> (np.ndarray, np.ndarray):
        """
        A utility method for computing the histogram of returns.

        :return: (np.ndarray, np.ndarray) A tuple of two np.ndarray, the first contains
        the limits of returns value of each bin of the histogram, and the second
        contains the bin's count value reported as density in relation to the counts
        in all other bins.
        """

        hist = [np.histogram(self.returns[:, i], bins=self.bins, density=False)
                for i in range(len(self.symbols_list))]
        counts = [np.expand_dims((h[0] / np.sum(h[0], axis=0, keepdims=True)), 1)
                  for h in hist]
        counts = np.concatenate(counts, 1)
        values = [np.expand_dims(h[1], 1) for h in hist]
        values = np.concatenate(values, 1)

        return values, counts

    def _compute_mean_annual_return(self) -> np.ndarray:
        """
        Utility method for computing the mean annual return of each asset, where the
        mean annual return is computed as the mean return over the entire
        requested period, multiplied by 255 (trading days per year on avg.)

        :return: (np.ndarray) The mean annual return for each asset
        """

        # Get mean returns
        mean_returns = np.mean(self.returns, axis=0)

        # Annualize
        mean_annual_returns = 255 * mean_returns

        return mean_annual_returns

    @property
    def mean_annual_return(self) -> np.ndarray:
        """
        Denotes the mean annual return of each asset, where the
        mean annual return is computed as the mean return over the entire
        requested period, multiplied by 255 (trading days per year on avg.)

        :return: (np.ndarray) The mean annual return for each asset
        """

        return self._compute_mean_annual_return()

    def _get_spectral_components(
            self, quotes: np.ndarray) -> None:
        """
        A utility method for computing the spectral components of an inputted signal
        and filter for those who passes a given threshold.

        :param quotes: (np.ndarray) The signal to analyze

        :return: None
        """

        assert not any(np.isnan(quotes).reshape(-1)), \
            "Detected NaNs in the 'quotes', please check the inputs."

        # Compute power spectrum and get spectral components
        frequencies, power_spectrum = welch(quotes, fs=1.0, window='hann',
                                            return_onesided=True, scaling='density',
                                            axis=0)
        total_energy = np.sum(power_spectrum, axis=0)
        normalized_power_spectrum = power_spectrum / np.expand_dims(total_energy, 0)

        assert not any(np.isnan(normalized_power_spectrum).reshape(-1)), \
            "Detected NaNs in the 'normalized_power_spectrum', please check the inputs."

        spectral_components = [
            np.where(normalized_power_spectrum[:, i] >
                     self.spectral_energy_threshold)[0]
            for i in range(self.n_assets)
        ]
        spectral_components_freqs = [
            frequencies[component]
            for component in spectral_components if len(component)
        ]
        spectral_components_energies = [
            normalized_power_spectrum[component, i]
            for i, component in enumerate(spectral_components) if len(component)
        ]

        assert len(spectral_components_freqs) == self.n_assets, \
            "Spectral components were not found for all assets, please try again" \
            "with a lower 'spectral_energy_threshold'."
        assert len(spectral_components_energies) == self.n_assets, \
            "Spectral components were not found for all assets, please try again" \
            "with a lower 'spectral_energy_threshold'."

        self.spectral_components_freqs = spectral_components_freqs
        self.spectral_components_energies = spectral_components_energies

    # TODO: Debug
    def generate_periodic_signal(self, quotes: np.ndarray,
                                 x_axis: np.ndarray) -> np.ndarray:
        """
        A utility method for generating a periodic signal based on a given raw signal
        and a temporal axis on which to calculate the signal

        :param quotes: (np.ndarray) The raw signal based on which to generate the signal
        :param x_axis: (np.ndarray) The temporal points for which to generate the signal
        (can be used also for future forecasting).

        :return: (np.ndarray) The computed periodic signal
        """

        assert (self.spectral_components_energies is not None and
                self.spectral_components_freqs is not None), \
            "Cannot call generate_periodic_signal without first calling " \
            "_get_spectral_components at least once, since otherwise we have no " \
            "spectral components based on which to compute the signal."

        periodic_signal = [
            (np.sum(
                np.concatenate(
                    [np.expand_dims((
                            self.spectral_components_energies[c][f] *
                            np.sin(2 * np.pi * freq * x_axis)),
                        1)
                        for f, freq in enumerate(component)],
                    1),
                axis=1) / len(component)) if len(component) else np.zeros_like(x_axis)
            for c, component in enumerate(self.spectral_components_freqs)
        ]

        # Concatenate all signals
        periodic_signal = np.concatenate([np.expand_dims(sig, 1)
                                          for sig in periodic_signal], 1)

        # Compute the offset of the periodic signal as the running mean of the
        # acutal raw signal
        sums = [
            np.mean(quotes[:, i:(i + self.trend_period_length)],
                    axis=1,
                    keepdims=True)
            for i in range(self.trend_period_length - 1)
        ]
        sums.extend(
            [
                np.mean(quotes[:, i:(i + self.trend_period_length)],
                        axis=1,
                        keepdims=True)
                for i in range((quotes.shape[1] - self.trend_period_length))
            ]
        )
        sums = np.concatenate(sums, 0)

        # Compute the amplitude of the periodic signal as the running std of the
        # actual raw signal
        sums_std = [
            np.std(quotes[:, i:(i + self.trend_period_length)], axis=1, keepdims=True)
            for i in range(self.trend_period_length - 1)
        ]
        sums_std.extend(
            [
                np.std(quotes[i:(i + (self.trend_period_length)), :], axis=0,
                       keepdims=True)
                for i in range((quotes.shape[1] - self.trend_period_length))
            ]
        )
        sums_std = np.concatenate(sums_std, 0)

        # Generate the finalized signal
        final_signal = (sums_std * periodic_signal) + sums

        return final_signal

    def _analyze_periodicty(self, quotes: np.ndarray) -> np.ndarray:
        """
        Performs a spectrum estimation for each asset using the Welch power-spectrum
        density estimator. Then detect every spectral component which relative energy is
        higher then the energy threshold component in the constructor.
        We then compose a new signal consistent of:
            S = sigma_{i = 1}^{n} E[i] * sin(2 * pi * Frequency[i])
        where E[i] is the energy of the spectral component i, and Frequency[i] is
        the frequency of the spectral component i.
        Finally, the periodic signal is taken to be:
            Y = S + Avg(x)
        Where Avg(x) refers to taking the running mean over the raw signal.

        :param quotes: (np.ndarray) The assets to analyze

        :return: (np.ndarray) The final periodic signal
        """

        self._get_spectral_components(quotes=quotes)

        # Compute periodic signal for each component
        x_axis = np.arange(quotes.shape[0] - 1)
        final_signal = self.generate_periodic_signal(quotes=quotes, x_axis=x_axis)

        return final_signal

    def _returns_emerging_trend(self, quotes: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        A utility method for detecting trends in assets' quotes over a recent, short
        time periods.

        :param quotes: (np.ndarray) The assets to analyze

        :return: (np.ndarray, np.ndarray) A tuple of two np.ndarray, the first contains
        the mean trends of each asset' returns over the most recent
        'trend_period_length' trading days, and the second element contains the return's
        standard deviation over the same period
        """

        period = quotes[-(self.trend_period_length + 1):, :]
        returns = self._compute_returns(period)
        trend_mean = np.mean(returns, axis=0)
        trend_std = np.std(returns, axis=0)

        return trend_mean, trend_std

    def _compute_overall_period_return(self) -> np.ndarray:
        """
        Utility method for computing the overall return of each asset over the most
        recent trading period, where the length of the period if determined by
        the 'trend_period_length' parameter given in the constructor.

        :return: (np.ndarray) The overall return of each asset of the specified period
        """

        # Get period
        recent_quotes = self.quotes[-(self.trend_period_length - 1):, :]

        # Compute overall returns
        overall_returns = ((recent_quotes[-1, :] - recent_quotes[0, :]) /
                           recent_quotes[0, :])

        return overall_returns

    @property
    def overall_period_return(self) -> np.ndarray:
        """
        Class property, denoting the overall return of each asset over the most
        recent trading period, where the length of the period if determined by
        the 'trend_period_length' parameter given in the constructor.

        :return: (np.ndarray) The overall return of each asset of the specified period
        """

        return self._compute_overall_period_return()

    def linear_regression_fit(self) -> np.ndarray:
        """
        A utility method which returns the slope, intercept and the R^2 values from
        performing a linear regression fit across the latest 'trend_period_length'
        trading periods.

        :return: A numpy array of shape (N, 3), containing the slope,
        intercept and the R^2 values of each one of the N asset, in that order.
        """

        x = np.arange(self.trend_period_length)
        results = [
            linregress(x, self.quotes[-self.trend_period_length:, i])
            for i in range(self.quotes.shape[1])
        ]
        results = np.concatenate(
            [
                np.expand_dims(np.array([res.slope, res.intercept, res.rvalue ** 2]), 0)
                for res in results
            ], 0
        )

        return results

    @property
    def top_k_performers(self) -> np.ndarray:
        """
        Class property, the top K performers

        :return: (np.ndarray) An array with shapes (# Assets), containing the
        the sorted indices of the top performers
        """

        argsort = np.argsort(self.cumulative_returns)[::-1]
        indices = np.zeros_like(argsort)
        for i, ind in enumerate(argsort):
            indices[ind] = i

        return indices

    @property
    def bottom_k_performers(self) -> np.ndarray:
        """
        Class property, the bottom K performers

        :return: (np.ndarray) An array with shapes (# Assets), containing the
        the sorted indices of the bottom performers
        """

        argsort = np.argsort(self.cumulative_returns)
        indices = np.zeros_like(argsort)
        for i, ind in enumerate(argsort):
            indices[ind] = i

        return indices

    def analyze(self, quotes: np.ndarray = None) -> dict:
        """
        The main method to use in the Analyzer class. It runs all currently available
        analysis  methods and returns per-assets results

        :param quotes: (np.ndarray) quotes to analyze, if None uses the quotes
        queried in the constructor.

        :return: (dict) a dictionary with the following key-value pairs:
        'sr': Sharpe-Ratio per-time point, per-asset
        'mean': mean annualized returns, per assets, over the entire period
        'recent_trend_mean': mean of returns, per assets over the most recent 'period'
        specified in the constructor.
        'recent_trend_std': std of returns, per assets over the most recent 'period'
        specified in the constructor.
        'periodicity': Linear combinations of sin functions based on Welch's
        power-spectrum estimation, which should fit each asset.
        'top_k': Top K average performers
        'bottom_k': Bottom K average performers
        """

        # Check inputs
        assert quotes is not None or self.quotes is not None, \
            f"Cannot pass quotes as None without also specifying querying " \
            f"parameters at construction time."

        if quotes is None:
            quotes = self.quotes

        # Perform analysis
        sr = self.sr

        # TODO: Fix
        # periodicity = self._analyze_periodicty(quotes=quotes)

        mean_annual_returns = self.mean_annual_return
        trend_mean, trend_std = self._returns_emerging_trend(quotes=quotes)
        linear_regression_fit = self.linear_regression_fit()
        top_k = self.top_k_performers
        bottom_k = self.bottom_k_performers

        analysis = {
            'sr': sr,
            'mean': mean_annual_returns,
            'recent_trend_mean': trend_mean,
            'recent_trend_std': trend_std,
            # 'periodicity': periodicity,
            'linear_regression_fit': linear_regression_fit,
            'top_k': top_k,
            'bottom_k': bottom_k,
        }

        return analysis
