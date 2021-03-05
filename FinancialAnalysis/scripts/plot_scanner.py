from analysis.analyzing import Analyzer
from FinancialAnalysis import PROJECT_ROOT
from FinancialAnalysis.analysis.scanning import Scanner
from FinancialAnalysis.visualizations.plot_assets import plot_assets_list
from FinancialAnalysis.stocks_io.data_queries import (
    get_sp500_symbols_wiki,
    get_nasdaq100_symbols_wiki,
)

import os

# Define data parameters
sp500_symbols, sp500_names = get_sp500_symbols_wiki()
nasdaq100_symbols, nasdaq100_names = get_nasdaq100_symbols_wiki()

symbols_list = tuple(sp500_symbols + nasdaq100_symbols)
start_date = "2016-02-25"
end_date = "2021-02-25"
quote_channel = 'Close'
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')

macro_criterions = {
    # 'five_years_div_yield': (2., 10.),
    'trailing_price2earnings': (5., 40.),
    'book2value_ratio': (0.5, 40.),
    # 'high_52w': (1.2, 1.5),
    # 'low_52w': (0.7, 0.9),
}
quote_criterions = {
    # 'sr': (2., 4.),
    'linear_regression_fit': (None, None, 0.7)
}

analyzer = Analyzer(
    symbols_list=symbols_list,
    start_date=start_date,
    end_date=end_date,
    quote_channel=quote_channel,
    adjust_prices=adjust_prices,
    cache_path=cache_path,
)
scanner = Scanner(
    symbols_list=symbols_list,
    start_date=start_date,
    end_date=end_date,
    quote_channel=quote_channel,
    adjust_prices=adjust_prices,
    cache_path=cache_path,
    analyzer=analyzer,
)

scanner.set_macro_criterions(macro_criterions)
scanner.set_quote_criterions(quote_criterions)

potential_assets = scanner.scan_for_potential_assets()
assert len(potential_assets)

potential_quotes = [scanner.quotes[:, i] for i, _ in potential_assets]
potential_macros = [scanner.macros[i] for i, _ in potential_assets]
potential_symbols = tuple([f"{sym}: {scanner.macros[i]['name']}" for i, sym in potential_assets])
plot_assets_list(
    assets_symbols=potential_symbols,
    assets_data=potential_quotes,
    dates=scanner.dates,
    assets_meta_data=potential_macros,
)
