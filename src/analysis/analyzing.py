from abc import ABC

import numpy as np


class Analyzer(ABC):
    """
    Class for performing time-series based analysis over an asset's quote.
    """

    def __init__(self):
        """
        Constructor method for the Analyzer class
        """

        self.time_series = None

    def set_quote(self, quote: np.ndarray) -> None:
        """
        A utility method for setting the time-series array to be analyzed.

        :param quote: (np.ndarray) The time-series on which the Analyzer class should
        operate

        :return: None
        """

        self.time_series = quote

    def _analyze_slope(self):
        pass

    def _analyze_periodicty(self):
        pass

    def _fit_distribution(self):
        pass


