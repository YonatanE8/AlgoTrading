from typing import Dict
from datetime import datetime
from FinancialAnalysis.utils.hashing import dict_hash

import os
import pickle
import requests
import bs4 as bs
import numpy as np
import yfinance as yf


def get_sp500_symbols_wiki(
        url: str = r'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers: Dict[str, str] = None) -> (list, list):
    """
    A method for getting the symbols of assets currently included as part
    of the S&P500 index.

    :param url: (str) Wiki url to query the symbols from. Defaults to
    'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'.
    :param headers: (dict) Defines the User-Agent for querying the website.
    Defaults to None, in which it then takes the value of:
    {'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                   ' AppleWebKit/537.17 (KHTML, like Gecko)'
                   ' Chrome/24.0.1312.27 Safari/537.17'}

    :return: (list, list) A list of symbols of stocks currently included
    in the S&P500 index, and a list of the companies names ordered similarly
    to the list of symbols.
    """

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                          ' AppleWebKit/537.17 (KHTML, like Gecko)'
                          ' Chrome/24.0.1312.27 Safari/537.17'
        }

    # Query the list of companies included in the S&P 500 index from Wikipedia
    resp = requests.get(url, headers=headers)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tables_rows = table.findAll('tr')[1:]
    symbols = [row.findAll('td')[0].text.strip() for row in tables_rows]
    names = [row.findAll('td')[1].text.strip() for row in tables_rows]

    return symbols, names


def get_nasdaq100_symbols_wiki(
        url: str = r'https://en.wikipedia.org/wiki/Nasdaq-100',
        headers: Dict[str, str] = None) -> (list, list):
    """
    A method for getting the symbols of assets currently included as part
    of the NASDAQ 100 index.

    :param url: (str) Wiki url to query the symbols from. Defaults to
    ''https://en.wikipedia.org/wiki/Nasdaq-100'.
    :param headers: (dict) Defines the User-Agent for querying the website.
    Defaults to None, in which it then takes the value of:
    {'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                   ' AppleWebKit/537.17 (KHTML, like Gecko)'
                   ' Chrome/24.0.1312.27 Safari/537.17'}

    :return: (list, list) A list of symbols of stocks currently included
    in the NASDAQ 100 index index, and a list of the companies names ordered similarly
    to the list of symbols.
    """

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                          ' AppleWebKit/537.17 (KHTML, like Gecko)'
                          ' Chrome/24.0.1312.27 Safari/537.17'
        }

    # Query the list of companies included in the S&P 500 index from Wikipedia
    resp = requests.get(url, headers=headers)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.findAll('table', {'class': 'wikitable sortable'})
    table = table[2]
    tables_rows = table.findAll('tr')[1:]
    symbols = [row.findAll('td')[1].text.strip() for row in tables_rows]
    names = [row.findAll('td')[0].text.strip() for row in tables_rows]

    return symbols, names


def get_nasdaq_listed_symbols(
        url: str = r"http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        headers: Dict[str, str] = None) -> (list, list):
    """
    A method for getting all of the symbols of assets traded at the NASDAQ
    stock exchange

    :param url: (str) NASDAQ url to query the symbols from. Defaults to
    'http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'.
    :param headers: (dict) Defines the User-Agent for querying the website.
    Defaults to None, in which it then takes the value of:
    {'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                   ' AppleWebKit/537.17 (KHTML, like Gecko)'
                   ' Chrome/24.0.1312.27 Safari/537.17'}

    :return: (list, list) A list of the symbols of all stocks currently
    traded in the NASDAQ, and a list of the companies names ordered similarly
    to the list of symbols.
    """

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                          ' AppleWebKit/537.17 (KHTML, like Gecko)'
                          ' Chrome/24.0.1312.27 Safari/537.17'
        }

    # Query data from NASDAQ Website
    response = requests.get(url, headers)
    txt_contents = response.text

    # Parse the symbols from the text
    lines = txt_contents.split(os.linesep)[1:-2]
    symbols = [line.split('|')[0] for line in lines]
    names = [line.split('|')[1].split('-')[0].strip(os.sep).strip(' ')
             for line in lines]

    return symbols, names


def _load_asset_data(symbol: str, start_date: str, end_date: str = None,
                     quote_channels: (str, ...) = ('Adj Close', ...),
                     adjust_prices: bool = True) -> (dict, dict):
    """
    Utility method, which actually gets the data related to a specific asset between
    a given date range. The returned data includes stock quotes between the given
    date range

    :param symbol: (str) The symbol of the stock for which data is to be queried
    :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
    :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
    If None uses today's date. Defualts to None.
    :param quote_channels: (Tuple) Tuple of strings, where each element should denote a
    quote channel to query stock prices by. The available channels are:
    'Close', 'Open', 'Low', 'High', 'Volume'.
    :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
    defaults to True.

    :return: (Tuple) Tuple containing two dicts:

    the first one containing all requested quotes, keyed by the requested
    'quote_channels', and 'Dates' which contains the temporal axis for example:
     'Close': NumPy array containing the Adj. Closing prices
     'Volume': NumPy array containing the trading volumes
     'Dates': NumPy array containing the trading volumes

    The second dictionary contains macro data for the asset, with the following keys:
    'name': Company name
    'sector': Company sector
    'beta': Volatility / Systematic Risk
    'dividend_rate': The total annual expected dividend payments
    'five_years_div_yield': Average annual dividend payments / stock price,
     averaged over the last 5 years.
    'trailing_price2earnings': Price to Earnings ratio, averaged over the last 12 months
    'trailing_price2sales': Price to Sales ratio, averaged over the last 12 months
    'book2value_ratio': Price to Book Value ratio, averaged over the last 12 months
    'profit_margins': Profit margins (Total Profit / Total Revenues)
    'high_52w': Highest market price in past 52 weeks
    'low_52w': Lowest market price in past 52 weeks
    'change_52w': Change in the asset market price over the past 52 weeks, in %.
    'last_dividend_date': Date in which the last dividend was paid
    'earnings_quarterly_growth': The amount by which the earnings in a quarter exceed
    the earnings in a corresponding quarter from a previous year, in %.
    """

    datetime_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, datetime_format)

    if end_date is None:
        end_date = datetime.today()

    else:
        end_date = datetime.strptime(end_date, datetime_format)

    # Initialize the ticker
    ticker = yf.Ticker(symbol)

    # Query macro data, some keys might be missing for some assets,
    # so query only those that exists
    info = ticker.info
    macro = {
        'name': info['shortName'] if 'shortName' in info else None,
        'sector': info['sector'] if 'sector' in info else None,
        'beta': info['beta'] if 'beta' in info else None,
        'dividend_rate': info['dividendRate'] if 'dividendRate' in info else None,
        'five_years_div_yield': (
            info['fiveYearAvgDividendYield'] if 'fiveYearAvgDividendYield' in info else None
        ),
        'trailing_price2earnings': info['trailingPE'] if 'trailingPE' in info else None,
        'trailing_price2sales': (
            info['priceToSalesTrailing12Months'] if 'priceToSalesTrailing12Months' in info else None
        ),
        'forward_price2earnings': info['forwardPE'] if 'forwardPE' in info else None,

        'price2book': info['priceToBook'] if 'priceToBook' in info else None,
        'profit_margins': info['profitMargins'] if 'profitMargins' in info else None,
        'high_52w': info['fiftyTwoWeekHigh'] if 'fiftyTwoWeekHigh' in info else None,
        'low_52w': info['fiftyTwoWeekLow'] if 'fiftyTwoWeekLow' in info else None,
        'change_52w': info['52WeekChange'] if '52WeekChange' in info else None,
        'last_dividend_date': (
            datetime.fromtimestamp(info['lastDividendDate'])
            if ('lastDividendDate' in info and isinstance(info['lastDividendDate'], int)) else
            None
        ),
        'earnings_quarterly_growth': (
            info['earningsQuarterlyGrowth'] if 'earningsQuarterlyGrowth' in info else None
        ),
        'yield': info['yield'] if 'yield' in info else None,
        'quarterly_revenue_growth': info['revenueQuarterlyGrowth'] if 'revenueQuarterlyGrowth' in info else None,
        'gross_margins': info['grossMargins'] if 'grossMargins' in info else None,
        'operating_margins': info['operatingMargins'] if 'operatingMargins' in info else None,
        'revenue_growth': info['revenueGrowth'] if 'revenueGrowth' in info else None,
        'analysts_recommendation': info['recommendationKey'] if 'recommendationKey' in info else None,
        'earnings_growth': info['earningsGrowth'] if 'earningsGrowth' in info else None,
        'roa': info['returnOnAssets'] if 'returnOnAssets' in info else None,
        'roe': info['returnOnEquity'] if 'returnOnEquity' in info else None,
        '3Y_beta': info['beta3Year'] if 'beta3Year' in info else None,
        '3Y_avg_return': info['threeYearAverageReturn'] if 'threeYearAverageReturn' in info else None,
        '5Y_avg_return': info['fiveYearAverageReturn'] if 'threeYearAverageReturn' in info else None,
        'debt2equity': info['debtToEquity'] if 'debtToEquity' in info else None,
        'quick_ratio': info['quickRatio'] if 'quickRatio' in info else None,
        'current_ratio': info['currentRatio'] if 'quickRatio' in info else None,
        'enterprise2ebitda': info['enterpriseToEbitda'] if 'enterpriseToEbitda' in info else None,
        'forward_eps': info['forwardEps'] if 'forwardEps' in info else None,
        'trailing_eps': info['trailingEps'] if 'trailingEps' in info else None,
        'short_ratio': info['shortRatio'] if 'shortRatio' in info else None,
        'shortPercentOfFloat': info['shortPercentOfFloat'] if 'shortPercentOfFloat' in info else None,
        'current_shorted_shares_ratio': (
            info['sharesShort'] / info['sharesOutstanding']
            if 'sharesShort' in info and 'sharesOutstanding' in info else
            None
        ),
        'total_cash_per_share': info['totalCashPerShare'] if 'totalCashPerShare' in info else None,
        'revenue_per_share': info['revenuePerShare'] if 'revenuePerShare' in info else None,
        'target_low_price_ratio': (
            info['targetLowPrice'] / info['currentPrice']
            if 'targetLowPrice' in info and 'currentPrice' in info else
            None
        ),
        'target_median_price_ratio': (
            info['targetMedianPrice'] / info['currentPrice']
            if 'targetMedianPrice' in info and 'currentPrice' in info else
            None
        ),
        'target_mean_price_ratio': (
            info['targetMeanPrice'] / info['currentPrice']
            if 'targetMeanPrice' in info and 'currentPrice' in info else
            None
        ),
        'percent_held_by_institutions': (
            info['heldPercentInstitutions'] if 'heldPercentInstitutions' in info else None
        ),
        'percent_held_by_insiders': (
            info['heldPercentInsiders'] if 'heldPercentInsiders' in info else None
        ),
        'price2earnings_growth_ratio': info['pegRatio'] if 'pegRatio' in info else None,
        'trailing_div_yield': info['trailingAnnualDividendYield'] if 'trailingAnnualDividendYield' in info else None,
        'trailing_div_rate': info['trailingAnnualDividendRate'] if 'trailingAnnualDividendRate' in info else None,
        'div_rate': info['dividendRate'] if 'dividendRate' in info else None,
        'payout_ratio': info['payoutRatio'] if 'payoutRatio' in info else None,
    }

    # Query historical quote prices
    quotes = ticker.history(start=start_date, end=end_date,
                            prepost=False, actions=False,
                            auto_adjust=adjust_prices, back_adjust=False,
                            rounding=False)

    # Get the temporal axis
    dates = [str(t).split('T')[0] for t in quotes[quote_channels[0]].index.values]

    # Get the financial quotes
    quotes = {channel: quotes[channel].values
              for channel in quote_channels}
    quotes['Dates'] = dates

    return quotes, macro


def get_asset_data(symbol: str, start_date: str, end_date: str = None,
                   quote_channels: (str, ...) = ('Adj Close', ...),
                   adjust_prices: bool = True,
                   cache_path: str = None) -> (dict, [dict, ...]):
    """
    Wrapper method around _load_asset_data, which either uses cached data if available,
    otherwise calls _load_asset_data in order to query data.

    :param symbol: (str) The symbol of the stock for which data is to be queried
    :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
    :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
    If None uses today's date. Defualts to None.
    :param quote_channels: (Tuple) Tuple of strings, where each element should denote a
    quote channel to query stock prices by. The available channels are:
    'Close', 'Open', 'Low', 'High', 'Volume'.
    :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
    defaults to True.
    :param cache_path: (str) Path to the directory in which to cache / look for cached
    data, if None does not use caching. Default is None.

    :return: (Tuple) Tuple containing two dicts:

    the first one containing all requested quotes, keyed by the requested
    'quote_channels', and 'Dates' which contains the temporal axis for example:
     'Close': NumPy array containing the Adj. Closing prices
     'Volume': NumPy array containing the trading volumes
     'Dates': NumPy array containing the trading volumes

    The second dictionary contains macro data for the asset, with the following keys:
    'name': Company name
    'sector': Company sector
    'beta': Volatility / Systematic Risk
    'dividend_rate': The total annual expected dividend payments
    'five_years_div_yield': Average annual dividend payments / stock price,
     averaged over the last 5 years.
    'trailing_price2earnings': Price to Earnings ratio, averaged over the last 12 months
    'trailing_price2sales': Price to Sales ratio, averaged over the last 12 months
    'book2value_ratio': Price to Book Value ratio, averaged over the last 12 months
    'profit_margins': Profit margins (Total Profit / Total Revenues)
    'high_52w': Highest market price in past 52 weeks
    'low_52w': Lowest market price in past 52 weeks
    'change_52w': Change in the asset market price over the past 52 weeks, in %.
    'last_dividend_date': Date in which the last dividend was paid
    'earnings_quarterly_growth': The amount by which the earnings in a quarter exceed
    the earnings in a corresponding quarter from a previous year, in %.
    """

    # Generate cache signature
    if cache_path is not None:
        input_dict = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'quote_channels': quote_channels,
            'adjust_prices': adjust_prices,
        }
        hash_signature = dict_hash(input_dict)
        data_file = os.path.join(cache_path, (hash_signature + '.pkl'))

        # Check if the data was already cached
        if os.path.isfile(data_file):
            with open(data_file, 'rb') as f:
                cached_data = pickle.load(f)
                quotes, macros = cached_data['quotes'], cached_data['macros']

        else:
            quotes, macros = _load_asset_data(symbol=symbol,
                                              start_date=start_date,
                                              end_date=end_date,
                                              quote_channels=quote_channels,
                                              adjust_prices=adjust_prices)

            with open(data_file, 'wb') as f:
                pickle.dump(obj={'quotes': quotes, 'macros': macros}, file=f)

    else:
        quotes, macros = _load_asset_data(symbol=symbol,
                                          start_date=start_date,
                                          end_date=end_date,
                                          quote_channels=quote_channels,
                                          adjust_prices=adjust_prices)

    return quotes, macros


def _load_multiple_assets(
        symbols_list: (str, ...),
        start_date: str,
        end_date: str = None,
        quote_channels: (str, ...) = ('Adj Close', ...),
        adjust_prices: bool = True,
        cache_path: str = None,
) -> (dict, [dict, ...], list):
    """
    A utility method for loading  N multiple assets,
    used by the get_multiple_assets method.

    :param symbols_list: (Tuple) A tuple with all of the symbols of the stocks
    for which data is to be queried.
    :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
    :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
    If None uses today's date. Defualts to None.
    :param quote_channels: (Tuple) Tuple of strings, where each element should denote a
    quote channel to query stock prices by. The available channels are:
    'Close', 'Open', 'Low', 'High', 'Volume'.
    :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
    defaults to True.

    :return: (Tuple) A tuple.

    The first element is a dictionary, keyed by the requested
    'quote_channels', and 'Dates' which contains the temporal axis for example.
    Each key contains a NumPy ndarray object of shape (N, T), where N is the
    number of assets and T is the number of trading days.

    The second element contains a list of N dicts, where each dict contains the
    macros of the respective asset in symbols_list/
    """

    # Load all requested assets
    quotes = []
    macros = []
    if cache_path is not None:
        cache_path = os.path.join(cache_path, 'single_assets')
        os.makedirs(cache_path, exist_ok=True)

    loaded_symbols = []
    for symbol in symbols_list:
        print(f"Loading data for {symbol}")

        # Generate cache signature
        if cache_path is not None:
            input_dict = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'quote_channels': quote_channels,
                'adjust_prices': adjust_prices,
            }
            hash_signature = dict_hash(input_dict)
            data_file = os.path.join(cache_path, (hash_signature + '.pkl'))

            # Check if the data was already cached
            if os.path.isfile(data_file):
                with open(data_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    quote, macro = cached_data['quote'], cached_data['macro']

                quotes.append(quote)
                macros.append(macro)
                loaded_symbols.append(symbol)

            else:
                try:
                    quote, macro = _load_asset_data(symbol=symbol,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    quote_channels=quote_channels,
                                                    adjust_prices=adjust_prices)

                except Exception as e:
                    print(f"Could not load the data for {symbol}, "
                          f"Exception is: {e}")
                    continue

                with open(data_file, 'wb') as f:
                    pickle.dump(obj={'quote': quote, 'macro': macro}, file=f)

                quotes.append(quote)
                macros.append(macro)
                loaded_symbols.append(symbol)

        else:
            try:
                quote, macro = _load_asset_data(symbol=symbol, start_date=start_date,
                                                end_date=end_date,
                                                quote_channels=quote_channels,
                                                adjust_prices=adjust_prices)

                quotes.append(quote)
                macros.append(macro)
                loaded_symbols.append(symbol)

            except:
                print(f"Could not load the data for {symbol}")

    # Concatenate the quotes NumPy arrays
    dates = [len(q['Dates']) for q in quotes]
    valid_dates_len = int(np.median(dates))
    valid_assets = [i for i, q in enumerate(quotes) if
                    len(q['Dates']) >= valid_dates_len]
    dates = quotes[valid_assets[0]]['Dates'][-valid_dates_len:]

    quotes = {
        channel: np.concatenate(
            [np.expand_dims(quotes[i][channel][-valid_dates_len:], 1) for i in
             valid_assets], 1)
        for channel in quote_channels if channel != 'Dates'
    }
    quotes['Dates'] = dates
    valid_symbols = [loaded_symbols[i] for i in valid_assets]
    macros = [macros[i] for i in valid_assets]

    return quotes, macros, valid_symbols


def get_multiple_assets(symbols_list: (str, ...), start_date: str, end_date: str = None,
                        quote_channels: (str, ...) = ('Adj Close', ...),
                        adjust_prices: bool = True,
                        cache_path: str = None) -> (dict, [dict, ...], list):
    """
    A method for querying N multiple assets and caching them if required.
    Wraps around the 'get_asset_data' method.

    :param symbols_list: (Tuple) A tuple with all of the symbols of the stocks
    for which data is to be queried.
    :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
    :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
    If None uses today's date. Defaults to None.
    :param quote_channels: (Tuple) Tuple of strings, where each element should denote a
    quote channel to query stock prices by. The available channels are:
    'Close', 'Open', 'Low', 'High', 'Volume'.
    :param adjust_prices: (bool) Whether to adjust the Close/Open/High/Low quotes,
    defaults to True.
    :param cache_path: (str) Path to the directory in which to cache / look for cached
    data, if None does not use caching. Default is None.

    :return: (Tuple) A tuple.

    The first element is a dictionary, keyed by the requested
    'quote_channels', and 'Dates' which contains the temporal axis for example.
    Each key contains a NumPy ndarray object of shape (N, T), where N is the
    number of assets and T is the number of trading days.

    The second element contains a list of N dicts, where each dict contains the
    macros of the respective asset in symbols_list/
    """

    # Generate cache signature
    if cache_path is not None:
        input_dict = {
            'symbols_list': symbols_list,
            'start_date': start_date,
            'end_date': end_date,
            'quote_channels': quote_channels,
            'adjust_prices': adjust_prices,
        }
        hash_signature = dict_hash(input_dict)
        data_file = os.path.join(cache_path, (hash_signature + '.pkl'))

        # Check if the data was already cached
        if os.path.isfile(data_file):
            with open(data_file, 'rb') as f:
                cached_data = pickle.load(f)
                quotes, macros, valid_symbols = (
                    cached_data['quotes'],
                    cached_data['macros'],
                    cached_data['valid_symbols'],
                )

        else:
            quotes, macros, valid_symbols = _load_multiple_assets(
                symbols_list=symbols_list,
                start_date=start_date,
                end_date=end_date,
                quote_channels=quote_channels,
                adjust_prices=adjust_prices,
                cache_path=cache_path,
            )

            with open(data_file, 'wb') as f:
                pickle.dump(
                    obj={
                        'quotes': quotes,
                        'macros': macros,
                        "valid_symbols": valid_symbols,
                    },
                    file=f,
                )

    else:
        quotes, macros, valid_symbols = _load_multiple_assets(
            symbols_list=symbols_list,
            start_date=start_date,
            end_date=end_date,
            quote_channels=quote_channels,
            adjust_prices=adjust_prices,
        )

    return quotes, macros, valid_symbols
