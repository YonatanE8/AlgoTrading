from src import PROJECT_ROOT
from src.analysis.scanning import Scanner
from src.stocks_io.data_queries import get_sp500_symbols_wiki

import os


# Define data parameters
symbols_list = ('MSFT', 'DIS', 'JPM', 'C', 'DAL')
start_date = "2015-12-03"
end_date = "2020-12-03"
quote_channels = ('Close',)
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')


scanner = Scanner(symbols_list=, start_date= end_date=,
                 quote_channel=, adjust_prices=,
                 smoother=, analyzer=,
                 cache_path=)