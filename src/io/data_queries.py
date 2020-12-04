from typing import Dict
from datetime import datetime
from src.utils.hashing import dict_hash

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


def get_asset_data(symbol: str, start_date: str, end_date: str = None,
                   quote_channels: (str, ...) = ('Adj Close', ...),
                   adjust_prices: bool = True) -> (dict, dict):
    """
    Get data related to a specific asset between a given date range.
    The returned data includes stock quotes between the given date range

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

    # Query macro data
    macro = {
        'name': ticker.info['shortName'],
        'sector': ticker.info['sector'],
        'beta': ticker.info['beta'],
        'dividend_rate': ticker.info['dividendRate'],
        'five_years_div_yield': ticker.info['fiveYearAvgDividendYield'],
        'trailing_price2earnings': ticker.info['trailingPE'],
        'trailing_price2sales': ticker.info['priceToSalesTrailing12Months'],
        'book2value_ratio': ticker.info['priceToBook'],
        'profit_margins': ticker.info['profitMargins'],
        'high_52w': ticker.info['fiftyTwoWeekHigh'],
        'low_52w': ticker.info['fiftyTwoWeekLow'],
        'change_52w': ticker.info['52WeekChange'],
        'last_dividend_date': datetime.fromtimestamp(ticker.info['lastDividendDate']),
        'earnings_quarterly_growth': ticker.info['earningsQuarterlyGrowth'],
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


def get_multiple_assets(symbols_list: (str, ...), start_date: str, end_date: str = None,
                        quote_channels: (str, ...) = ('Adj Close', ...),
                        adjust_prices: bool = True,
                        cache_path: str = None) -> (dict, [dict, ...]):
    """
    A method for querying N multiple assets and caching them if required.
    Wraps around the 'get_asset_data' method.

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
    input_dict = {
        'symbols_list': symbols_list,
        'start_date': start_date,
        'end_date': end_date,
        'quote_channels': quote_channels,
        'adjust_prices': adjust_prices,
    }
    hash_signature = dict_hash(input_dict)
    data_file = os.path.join(cache_path, (hash_signature + '.pkl'))

    if cache_path is not None:
        # Check if the data was already cached
        if os.path.isfile(data_file):
            with open(data_file, 'rb') as f:
                cached_data = pickle.load(f)
                quotes, macros = cached_data['quotes'], cached_data['macros']

    else:
        # Load all requested assets
        quotes = []
        macros = []
        for symbol in symbols_list:
            print(f"Fetching Data for {symbol}")

            try:
                quote, macro = get_asset_data(symbol=symbol, start_date=start_date,
                                              end_date=end_date,
                                              quote_channels=quote_channels,
                                              adjust_prices=adjust_prices)
                quotes.append(quote)
                macros.append(macro)

            except ValueError:
                print(f"Could not load the data for {symbol}")

        # Concatenate the quotes NumPy arrays
        quotes = {
            channel: np.concatenate([np.expand_dims(q[channel], 1) for q in quotes], 1)
            for channel in quote_channels
        }

        if cache_path is not None:
            with open(data_file, 'wb') as f:
                pickle.dump(obj={
                    'quotes': quotes,
                    'macros': macros,
                }, file=f)

    return quotes, macros



