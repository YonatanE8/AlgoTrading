from FinancialAnalysis.analysis.analyzing import Analyzer
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
start_date = "2017-06-03"
end_date = "2022-06-03"
quote_channel = 'Close'
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')

macro_criterions = {
    'five_years_div_yield': (0.1, 10.),
    'trailing_price2earnings': (0., 20.),
    # 'book2value_ratio': (0.5, 15.),
    # 'high_52w': (1.5, 2.0),
    # 'low_52w': (0.6, 0.8),
    'operating_margins': (0.2, 1.),
    'quick_ratio': (1.5, 5.),
    'current_ratio': (1.5, 5.),
    'enterprise2ebitda': (1., 10),
    'short_ratio': (0., 5.),
    'target_mean_price_ratio': (1.1, 2.),
    'percent_held_by_insiders': (0.1, 1),
    'price2earnings_growth_ratio': (0., 2.),
}
quote_criterions = {
    # 'sr': (2., 4.),
    # 'bottom_k': 75,
    # 'top_k': 250,
    # 'linear_regression_fit': (None, None, 0.8)
}

trend_period_length = 60
analyzer = Analyzer(
    symbols_list=symbols_list,
    start_date=start_date,
    end_date=end_date,
    quote_channel=quote_channel,
    adjust_prices=adjust_prices,
    trend_period_length=trend_period_length,
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
# scanner.set_quote_criterions(quote_criterions)

ignore_none = False
potential_assets = scanner.scan_for_potential_assets(ignore_none=ignore_none)
assert len(potential_assets)

potential_quotes = [scanner.quotes[:, i] for i, _ in potential_assets]
potential_macros = [scanner.macros[i] for i, _ in potential_assets]
potential_symbols = tuple([f"{sym}: {scanner.macros[i]['name']}" for i, sym in potential_assets])
plot_assets_list(
    assets_symbols=potential_symbols,
    assets_data=potential_quotes,
    dates=scanner.dates,
    assets_meta_data=potential_macros,
    display_meta_paramets=(
        'dividend_rate',
        'five_years_div_yield',
        'trailing_price2earnings',
    ),
)
