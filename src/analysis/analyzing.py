from abc import ABC
from scipy.signal import welch
from src.io.data_queries import get_asset_data, get_multiple_assets

import numpy as np


class Analyzer(ABC):
    """
    Class for performing time-series based analysis over an asset's quote.
    """

    def __init__(self, symbols_list: tuple, start_date: str, end_date: str = None,
                 quote_channel: str = 'Close', adjust_prices: bool = True,
                 risk_free_asset_symbol: str = '^IRX', bins: int = 10,
                 spectral_energy_threshold: float = 0.2, trend_period_length: int = 22,
                 cache_path: str = None):
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
        self.n_assets = len(symbols_list)
        self.symbols_list = symbols_list
        self.start_date = start_date
        self.end_date = end_date
        self.quote_channel = quote_channel
        self.adjust_prices = adjust_prices
        self.risk_free_asset_symbol = risk_free_asset_symbol
        self.bins = bins
        self.spectral_energy_threshold = spectral_energy_threshold
        self.trend_period_length = trend_period_length

        # Query assets
        quotes = get_multiple_assets(symbols_list=symbols_list, start_date=start_date,
                                     end_date=end_date,
                                     quote_channels=(quote_channel,),
                                     adjust_prices=adjust_prices,
                                     cache_path=cache_path)
        self.quotes = quotes[quote_channel]
        self.returns = self._compute_returns(self.quotes)

        # Query the risk free asset
        risk_free_asset = get_asset_data(symbol=risk_free_asset_symbol,
                                         start_date=start_date, end_date=end_date,
                                         quote_channels=(quote_channel,),
                                         adjust_prices=adjust_prices,
                                         cache_path=cache_path)
        self.risk_free_asset = risk_free_asset[quote_channel]

        # Risk-free asset is already reported as returns, remove the first entry so that
        # the risk-free returns will align with the assets' returns
        self.risk_free_returns = self.risk_free_asset[1:]

    def _compute_returns(self, quotes: np.ndarray) -> np.ndarray:
        """
        Utility method for computing returns

        :param quotes: (np.ndarray) The quotes for which to compute returns.

        :return: (np.ndarray) Returns of all assets
        """

        returns = np.diff(quotes, axis=0) / quotes[:-1]

        return returns

    def _analyze_sr(self) -> np.ndarray:
        """
        Utility method for computing the assets Sharpe-Ratio

        :return: (np.ndarray) Sharpe-Ratio of all specified assets over the period
        specified in the constructor.
        """

        # Compute the excess returns and standard-deviation of the excess returns
        excess_returns = self.returns - np.expand_dims(self.risk_free_returns, 1)

        n_samples = excess_returns.shape[0] - 1
        excess_returns_stds = [np.std(excess_returns[:i + 1, :], axis=0, keepdims=True)
                               for i in range(1, n_samples)]
        excess_returns_stds = np.concatenate(excess_returns_stds, axis=0)

        # Compute SR
        sr = excess_returns / excess_returns_stds

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
        contains the bin's value count.
        """

        counts, values = np.histogram(self.returns, bins=self.bins, density=True)

        return values, counts

    @property
    def mean_annual_return(self) -> float:
        """
        Class property, denoting the mean annual return of each asset, where the
        mean annual return is computed as the mean return over the entire
        requested period, multiplied by 255 (trading days per year on avg.)

        :return: (float) The mean annual return
        """

        # Get mean returns
        mean_returns = np.mean(self.returns, axis=0)

        # Annualize
        mean_annual_returns = 255 * mean_returns

        return mean_annual_returns

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

        # Compute power spectrum and get spectral components
        frequencies, power_spectrum = welch(quotes, fs=1.0, window='hann',
                                            return_onesided=True, scaling='density',
                                            axis=0)
        total_energy = np.sum(power_spectrum, axis=0)
        normalized_power_spectrum = power_spectrum / np.expand_dims(total_energy, 0)

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

        # Copmute periodic signal for each component
        x_axis = np.arange(len(quotes.shape[0]))
        periodic_signal = [
            (np.sum(
                np.concatenate(
                    [np.expand_dims((
                            spectral_components_energies[c] *
                            np.sin(2 * np.pi * freq * x_axis)),
                        1)
                        for freq in component],
                    1),
                axis=1) / len(component)) if len(component) else np.zeros_like(x_axis)
            for c, component in enumerate(spectral_components_freqs)
        ]

        # Concatenate all signals
        periodic_signal = np.concatenate([np.expand_dims(sig, 1)
                                          for sig in periodic_signal], 1)

        # Add the running average window
        sums = np.cumsum(quotes, axis=0)
        sums = [np.expand_dims(sums[i, :], 0) / (i + 1) for i in range(quotes.shape[0])]
        sums = np.concatenate(sums, 0)

        final_signal = periodic_signal + sums

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

        period = quotes[-self.trend_period_length:, :]
        returns = self._compute_returns(period)
        trend_mean = np.mean(returns, axis=0)
        trend_std = np.std(returns, axis=0)

        return trend_mean, trend_std




