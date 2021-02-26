from src import PROJECT_ROOT
from src.analysis.scanning import Scanner
from src.visualizations.plot_assets import plot_assets_list
from src.stocks_io.data_queries import get_sp500_symbols_wiki, get_nasdaq_listed_symbols, get_multiple_assets

import os

# Define data parameters
sp500_symbols, sp500_names = get_sp500_symbols_wiki()
nasdaq100_symbols, nasdaq100_names = get_nasdaq_listed_symbols()

symbols_list = tuple(sp500_symbols + nasdaq100_symbols)
start_date = "2016-02-25"
end_date = "2021-02-25"
quote_channel = 'Close'
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')

macro_criterions = {
    'fiveYearAvgDividendYield': (1., 10.),
    'trailing_price2earnings': (5., 35.),
    'book2value_ratio': (0.5, 30.),
    'high_52w': (1.2, 1.5),
}
quote_criterions = {
    'sr': (2., 4.),
}

scanner = Scanner(
    symbols_list=symbols_list,
    start_date=start_date,
    end_date=end_date,
    quote_channel=quote_channel,
    adjust_prices=adjust_prices,
    cache_path=cache_path,
)

scanner.set_macro_criterions(macro_criterions)
scanner.set_quote_criterions(quote_criterions)

potential_assets = scanner.scan_for_potential_assets()

quotes, macros = get_multiple_assets(
    symbols_list=potential_assets,
    start_date=start_date,
    end_date=end_date,
    cache_path=cache_path,
)

plot_assets_list(
    assets_symbols=tuple(potential_assets),
    assets_data=quotes,
    dates=quotes['Dates'],
    assets_meta_data=macros,
)
