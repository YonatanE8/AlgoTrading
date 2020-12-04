from typing import Dict


import os
import pickle
import requests
import bs4 as bs
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

from datetime import datetime
from edgar import Edgar, Company
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


def get_quotes(symbol: str, start_date: str, end_date: str = None,
               data_source: str = 'yahoo') -> dict:
    """
    Get stock quotes between a date range.

    :param symbol: (str) The symbol of the stock for which data is to be queried
    :param start_date: (str) Starting date, should be formatted as 'year-month-day'".
    :param end_date: (str) Ending date, should be formatted as 'year-month-day'".
    If None uses today's date. Defualts to None.
    :param data_source: (str) Either 'yahoo' or 'pandas'.
     Specify the source from which to query the data from. Defaults to yahoo.

    :return:
    """

    assert data_source in ('yahoo', 'pandas'), \
        f"{data_source} must be either 'yahoo' or 'pandas'"

    datetime_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, datetime_format)

    if end_date is None:
        end_date = datetime.today()

    else:
        end_date = datetime.strptime(end_date, datetime_format)

    meta_data = yf.Ticker(symbol)

    try:
        data_reader = web.DataReader(name=symbol, data_source=data_source,
                                     start=start_date, end=end_date)

    except:
        print(f"Could not load data for {symbol}")

    quote = {
        'PB': price_to_book,
        'PE': price_to_earnings,
        'DivYield': dividend_yield,
        '1YearDivRate': one_yr_dividend_rate,
        '5YearsAvgDivYield': five_yr_dividend_rate,
        'ProfitMargins': profit_margins
    }

    return quote


def get_edgar_fillings(company_name: str, filling_type: str = None,
                       cik_num:  str = None, no_of_documents: int = 10):
    """

    :param company_name:
    :param filling_type:
    :param cik_num:
    :param no_of_documents:
    :return:
    """

    if cik_num is None:
        edgar = Edgar()
        company_name = edgar.find_company_name(company_name)[0]
        cik_num = edgar.get_cik_by_company_name(company_name)

        company = Company(company_name, cik_num)

        if filling_type is not None:
            tree = company.get_all_filings(filing_type=filling_type)

        else:
            tree = company.get_all_filings()

        docs = company.get_documents(tree=tree, no_of_documents=no_of_documents)
        info = company.get_company_info()

    else:
        company = Company(company_name, cik_num)

        if filling_type is not None:
            tree = company.get_all_filings(filing_type=filling_type)

        else:
            tree = company.get_all_filings()

        docs = company.get_documents(tree=tree, no_of_documents=no_of_documents)
        info = company.get_company_info()

    return tree, docs, info


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



