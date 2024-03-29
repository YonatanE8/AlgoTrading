from abc import ABC
from typing import List, Tuple
from FinancialAnalysis.analysis.smoothing import Smoother
from FinancialAnalysis.analysis.analyzing import Analyzer
from FinancialAnalysis.stocks_io.data_queries import get_multiple_assets


class Scanner(ABC):
    """
    Class for performing scans over a large number of assets, using pre-determined
    macro_criterions in order to automatically detect promising assets for investment.
    """

    def __init__(
            self,
            symbols_list: (str, ...),
            start_date: str,
            end_date: str,
            quote_channel: str,
            adjust_prices: bool = True,
            smoother: Smoother = None,
            analyzer: Analyzer = None,
            cache_path: str = None,
    ):
        """
        Constructor method for the scanner object.

        :param symbols_list: (Tuple) A tuple containing the listed symbols of the assets
        to be considered.
        :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
        :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
        If None uses today's date. Defaults to None.
        :param quote_channel: (str) The quote channel perform analysis by.
        The available channels are: 'Close', 'Open', 'Low', 'High', 'Volume'.
        :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
        defaults to True.
        :param smoother: (Smoother) An instantiated Smoother object to be used.
        If None, no smoothing is applied. Defaults to None.
        :param analyzer: (Analyzer) An instantiated Analyzer object to be used for
        quote based criterions. If None, no quote based criterions can be applied.
         Defaults to None.
        :param cache_path: (str) Path to the directory in which to cache / look for
        cached data, if None does not use caching. Default is None.
        """

        self.start_date = start_date
        self.quote_channel = quote_channel
        self.adjust_prices = adjust_prices
        self.cache_path = cache_path
        self._analyzer = analyzer

        quotes, self.macros, self.symbols_list = get_multiple_assets(
            symbols_list=symbols_list,
            start_date=start_date,
            end_date=end_date,
            quote_channels=(quote_channel,),
            adjust_prices=adjust_prices,
            cache_path=cache_path,
        )

        self.dates = quotes['Dates']
        self.quotes = quotes[quote_channel]
        self._smoother = smoother

        # Set place-holders
        self.relevant_assets = []
        self._viable_macro_criterions = (
            'sector',
            'beta',
            'dividend_rate',
            'five_years_div_yield',
            'fiveYearAvgDividendYield',
            'trailing_price2earnings',
            'trailing_price2sales',
            'book2value_ratio',
            'profit_margins',
            'high_52w',
            'low_52w',
            'change_52w',
            'last_dividend_date',
            'earnings_quarterly_growth',
            'yield',
            'quarterly_revenue_growth',
            'gross_margins',
            'operating_margins',
            'revenue_growth',
            'analysts_recommendation',
            'earnings_growth',
            'roa',
            'roe',
            '3Y_beta',
            '3Y_avg_return',
            '5Y_avg_return',
            'debt2equity',
            'quick_ratio',
            'current_ratio',
            'enterprise2ebitda',
            'forward_eps',
            'trailing_eps',
            'short_ratio',
            'shortPercentOfFloat',
            'current_shorted_shares_ratio',
            'total_cash_per_share',
            'revenue_per_share',
            'target_low_price_ratio',
            'target_median_price_ratio',
            'target_mean_price_ratio',
            'percent_held_by_institutions',
            'percent_held_by_insiders',
            'price2earnings_growth_ratio',
            'trailing_div_yield',
            'trailing_div_rate',
            'div_rate',
            'payout_ratio',
        )
        self.macro_criterions = {}

        self._viable_quote_criterions = [
            'sr',
            'mean',
            'recent_trend_mean',
            'recent_trend_std',
            'overall_period_return',
            'linear_regression_fit',
            'top_k',
            'bottom_k',
        ]
        self.quote_criterions = {}

    def set_smoother(self, smoother: Smoother) -> None:
        """
        A method for setting a Smoother object, enabling the analysis to be performed
        over a smoothed signal instead of over the raw signal itself.

        :param smoother: (Smoother) An instantiated smoother object to be used.

        :return: None
        """

        self._smoother = smoother

    def set_analyzer(self, analyzer: Analyzer) -> None:
        """
        A method for setting a Smoother object, enabling the analysis to be performed
        over a smoothed signal instead of over the raw signal itself.

        :param analyzer: (Analyzer) An instantiated Analyzer object to be used for
        quote based criterions.

        :return: None
        """

        self._analyzer = analyzer

    @property
    def viable_macro_criterions(self) -> (str, ...):
        """
        A class property, returns the list of viable macro criterions to be considered.

        :return: (tuple) A tuple containing a list of all viable macro criterions
         to be considered.
        """

        return self._viable_macro_criterions

    @property
    def viable_quotes_criterions(self) -> (str, ...):
        """
        A class property, returns the list of viable quote criterions to be considered.

        :return: (tuple) A tuple containing a list of all viable quote criterions
         to be considered.
        """

        return self._viable_quote_criterions

    def set_macro_criterions(self, criterions: dict) -> None:
        """
        A method used for specifying criterions that should hold over the macro
        characteristics of an asset,
        in order for it to be considered as eligible for investment.

        :param criterions: (dict) A dictionary specifying each criterion as a key-value
        pair, where the key is the name of the relevant macro field, and the value is
        a collection or range of acceptable values, i.e. the value for each key must
        be an iterable object.
        If the field is not specified in the queried data for a specific asset,
        analysis for that field, for that asset, is not performed.

        For example, one can specify the following criterions (for a full list of
        viable criterions please check the viable_macro_criterions property.

        criterions = {
            'sector': ('Health Care', 'Financials', 'Communication Services'),
            'beta': (minVal, maxVal),
            'dividend_rate': (minVal, maxVal),
            'trailing_price2earnings': (minVal, maxVal),
            'high_52w': (minVal in % (relative to current price)),
                         maxVal in % (relative to current price)),
            'low_52w': (minVal in % (relative to current price),
                         maxVal in % (relative to current price)),
            'earningsQuarterlyGrowth': (minVal in %, maxVal in % ),
        }


        :return: None
        """

        for key in criterions:
            assert key in self._viable_macro_criterions, \
                f"{key} is not a valid macro criterion, " \
                f"please refer to the viable_macro_criterions property" \
                f" for all viable criterions."

        self.macro_criterions.update(criterions)

    def _test_macro_criterion(self, asset_macro: dict, current_price: float = None,
                              ignore_none: bool = False) -> {bool, None}:
        """
        A utility method for testing whether a specific asset upholds the macro
        requirements specified in self.macro_criterions.

        :param asset_macro: (dict) dict containing all of the macro information for
        the asset to be considered.
        :param current_price: (float) Used with the relative fields: 'high_52w',
        'low_52w'. Must be specified if those fields are specified in
        self.macro_criterions.
        :param ignore_none: (bool) Whether to return None if encountered a missing
        value in asset_macro for any criterion. If False, returns None, if True,
        ignores the missing field and continue as usual. Default is False.

        :return: (bool/None) True/False/None, indicating whether the asset upholds all
        required criterions. Returns True if yes, False if not, and None, if some
        required macro information is missing and the test cannot be performed and
        ignore_none == False.
        """

        relative_fields = ('high_52w', 'low_52w')
        apply_relative_fields = False
        for field in relative_fields:
            if field in self.macro_criterions:
                apply_relative_fields = True

        assert current_price is not None or not apply_relative_fields, \
            "'current_price' must be specified if either " \
            "'high_52w' or 'low_52w' are specified in self.macro_criterions"

        for criterion in self.macro_criterions:
            if criterion not in asset_macro or asset_macro[criterion] is None:
                if not ignore_none:
                    return None

            elif isinstance(asset_macro[criterion], str):
                if criterion not in asset_macro or asset_macro[criterion] not in \
                        self.macro_criterions[criterion]:
                    return False

            elif not isinstance(asset_macro[criterion], str):
                if (
                        criterion not in asset_macro or
                        criterion in relative_fields and
                        (
                                (asset_macro[criterion] / current_price) <
                                self.macro_criterions[criterion][0] or
                                (asset_macro[criterion] / current_price) >
                                self.macro_criterions[criterion][1]
                        )
                ):
                    return False

                elif (
                        criterion not in asset_macro or
                        (
                                criterion not in relative_fields and
                                (
                                        asset_macro[criterion] < self.macro_criterions[criterion][0] or
                                        asset_macro[criterion] > self.macro_criterions[criterion][1]
                                )
                        )
                ):
                    return False

        return True

    def set_quote_criterions(self, criterions: dict) -> None:
        """
        A method used for specifying criterions that should hold over the quotes signal
        of an asset in order for it to be considered as eligible for investment.

        :param criterions: (dict) A dictionary specifying each criterion as a key-value
        pair, where the key is the name of the relevant analysis to apply to the quotes,
        and the value is a collection or range of acceptable values, i.e.
        the value for each key must be an iterable object.

        For example, one can specify the following criterions.
        Viable criterions are:
        'sr': (minVal, maxVal),
        'mean': (minVal, maxVal),
        'recent_trend_mean': (minVal, maxVal),
        'recent_trend_std': (minVal, maxVal),

        :return: None
        """

        for key in criterions:
            assert key in self._viable_quote_criterions, \
                f"{key} is not a valid quote criterion, " \
                f"please refer to the viable_quote_criterions property" \
                f" for all viable criterions."

        self.quote_criterions.update(criterions)

    def _test_quote_criterion(self, asset_quote_stats: dict) -> bool:
        """
        A utility method for testing whether a specific asset upholds the quote
        requirements specified in self.quote_criterions.

        :param asset_quote_stats:  (dict) A dictionary containing the results of all
        quotes based analysis for the asset in question.

        :return: (bool) True/False, indicating whether the asset upholds all
        required criterions. Returns True if yes, False if not.
        """

        for criterion in self.quote_criterions:
            if criterion == 'linear_regression_fit':
                assert len(self.quote_criterions[criterion]) == 3

                slope = asset_quote_stats[criterion][0]
                intercept = asset_quote_stats[criterion][1]
                r_sq = asset_quote_stats[criterion][2]

                slope_criterion = self.quote_criterions[criterion][0]
                intercept_criterion = self.quote_criterions[criterion][1]
                r_sq_criterion = self.quote_criterions[criterion][2]

                cond = True
                if slope_criterion is not None and slope < slope_criterion:
                    cond = False

                if intercept_criterion is not None and intercept < intercept_criterion:
                    cond = False

                if r_sq_criterion is not None and r_sq < r_sq_criterion:
                    cond = False

                return cond

            elif criterion == 'top_k' or criterion == 'bottom_k':
                if asset_quote_stats[criterion] < self.quote_criterions[criterion]:
                    return True

                else:
                    return False

            elif (asset_quote_stats[criterion] < self.quote_criterions[criterion][0] or
                  asset_quote_stats[criterion] > self.quote_criterions[criterion][
                      1]):
                return False

        return True

    def scan_for_potential_assets(self, ignore_none: bool = True) -> List[Tuple[int, str]]:
        """
        The main method to be used by a user in the Scanner class. After all macro &
        quotes criterions have been specified, scan all assets given in construction
        over the specified temporal period, and detect all asset that fit the
        required criterions.

        :param ignore_none: (bool) Whether to ignore Missing macro values or not. Default is True.

        :return: (list) A list of string, containing the trade symbols of all suitable
        assets.
        """

        if self._analyzer is not None:
            quotes_analysis = self._analyzer.analyze()

        else:
            quotes_analysis = None

        symbols = []
        for i, symbol in enumerate(self.symbols_list):
            macro = self.macros[i]
            quotes_criterions = (
                {
                    key: quotes_analysis[key][i]
                    for key in quotes_analysis
                }
                if quotes_analysis is not None
                else None
            )

            macro_criterion = self._test_macro_criterion(
                asset_macro=macro, current_price=self.quotes[-1, i], ignore_none=ignore_none)

            quote_criterion = (
                self._test_quote_criterion(asset_quote_stats=quotes_criterions)
                if quotes_criterions is not None else True
            )

            if quote_criterion and macro_criterion:
                symbols.append((i, symbol))

        return symbols
