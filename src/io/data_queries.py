from typing import Dict


import os
import pickle
import requests
import bs4 as bs
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta


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
    'quote_channels', and 'dates' which contains the temporal axis for example:
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


def get_sp500_assets(analysis_period_len: int = 1825, data_source: str = 'yahoo',
                     metric2_analyze: str = 'Close', start_date: str = None):
    """

    :param analysis_period_len:
    :param data_source:
    :param metric2_analyze:
    :param start_date:
    :return:
    """

    # Get all tickers
    sp500_tickers, sp500_names = get_sp500_symbols_wiki()

    # Get period to analyze
    if start_date is None:
        today = datetime.today()

    else:
        today = datetime.strptime(start_date, "%Y-%m-%d")

    yesterday = today - timedelta(days=1)
    first_day = yesterday - timedelta(days=analysis_period_len)

    yesterday = str(yesterday).split(' ')[0]
    first_day = str(first_day).split(' ')[0]

    start_date = datetime.strptime(first_day, "%Y-%m-%d")
    end_date = datetime.strptime(yesterday, "%Y-%m-%d")

    data_file = os.path.join(DATA_DIR,
                             f"{str(start_date).split()[0].replace('-', '_')}_to_"
                             f"{str(end_date).split()[0].replace('-', '_')}_"
                             f"{metric2_analyze}.pkl")

    if os.path.isfile(data_file):
        with open(data_file, 'rb') as f:
            res_dict = pickle.load(f)

            sp500_tickers = res_dict['sp500_tickers']
            tickers_data = res_dict['tickers_data']
            tickers_meta_data = res_dict['tickers_meta_data']

    else:
        if data_source == 'quandl':
            api_token = quandl_api_toekn

        elif data_source == 'iex':
            api_token = iex_api_token

        else:
            api_token = None

        # Get the SP500 data
        tickers_data = []
        tickers_meta_data = []
        for ticker in sp500_tickers:
            print(f"Fetching Data for {ticker}")
            try:
                ticker_qoute = get_quotes(symbol=ticker, start_date=start_date,
                                          end_date=end_date, data_source=data_source,
                                          api_key=api_token)

                tickers_data.append(np.expand_dims(
                    ticker_qoute[0][metric2_analyze].values, 1))
                tickers_meta_data.append(ticker_qoute[1])

            except:
                print(f"Could not load the data for {ticker}")

        n_days = [t.shape[0] for t in tickers_data]
        max_days = np.max(n_days)
        relevant_tickers = [i for i in range(len(tickers_data))
                            if tickers_data[i].shape[0] == max_days]
        tickers_data = [tickers_data[i] for i in relevant_tickers]
        sp500_tickers = [sp500_tickers[i] for i in relevant_tickers]
        tickers_meta_data = [tickers_meta_data[i] for i in relevant_tickers]
        tickers_data = np.concatenate(tickers_data, 1)

        res_dict = {
            'sp500_tickers': sp500_tickers,
            'tickers_data': tickers_data,
            'tickers_meta_data': tickers_meta_data,
        }

        with open(data_file, 'wb') as f:
            pickle.dump(obj=res_dict, file=f)

    return sp500_tickers, tickers_data, tickers_meta_data, sp500_names



