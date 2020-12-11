from abc import ABC
from src.io.data_queries import get_asset_data, get_multiple_assets

import numpy as np


class Analyzer(ABC):
    """
    Class for performing time-series based analysis over an asset's quote.
    """

    def __init__(self, symbols_list: tuple, start_date: str, end_date: str = None,
                 quote_channel: str = 'Close', adjust_prices: bool = True,
                 risk_free_asset_symbol: str = '^IRX', bins: int = 10,
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
        :param cache_path: (str) Path to the directory in which to cache / look for
        cached data, if None does not use caching. Default is None.
        """

        # Setup
        self.symbols_list = symbols_list
        self.start_date = start_date
        self.end_date = end_date
        self.quote_channel = quote_channel
        self.adjust_prices = adjust_prices
        self.risk_free_asset_symbol = risk_free_asset_symbol
        self.bins = bins

        # Query assets
        quotes = get_multiple_assets(symbols_list=symbols_list, start_date=start_date,
                                     end_date=end_date,
                                     quote_channels=(quote_channel, ),
                                     adjust_prices=adjust_prices,
                                     cache_path=cache_path)
        self.quotes = quotes[quote_channel]
        self.returns = self._compute_returns(self.quotes)

        # Query the risk free asset
        risk_free_asset = get_asset_data(symbol=risk_free_asset_symbol,
                                         start_date=start_date, end_date=end_date,
                                         quote_channels=(quote_channel, ),
                                         adjust_prices=adjust_prices,
                                         cache_path=cache_path)
        self.risk_free_asset = risk_free_asset[quote_channel]
        self.risk_free_returns = self._compute_returns(self.risk_free_asset)

    def _compute_returns(self, quotes: np.ndarray) -> np.ndarray:
        """
        Utility method for computing returns

        :param quotes: (np.ndarray) The quotes for which to compute returns.

        :return: (np.ndarray) Returns of all assets
        """

        returns = np.diff(quotes, axis=0)

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

    def _returns_histogram(self) -> (np.ndarray, np.ndarray):
        """
        A utility method for computing the histogram of returns.

        :return: (np.ndarray, np.ndarray) A tuple of two np.ndarray, the first contains
        the limits of returns value of each bin of the histogram, and the second
        contains the bin's value count.
        """

        counts, values = np.histogram(self.returns, bins=self.bins, density=True)

        return values, counts

    def _analyze_periodicty(self):
        pass



