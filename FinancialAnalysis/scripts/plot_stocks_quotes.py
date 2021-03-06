from FinancialAnalysis import PROJECT_ROOT
from FinancialAnalysis.stocks_io.data_queries import get_multiple_assets
from FinancialAnalysis.visualizations.plot_assets import plot_assets_list

import os

# symbols_list = ('MSFT', 'DIS', 'JPM', 'C', 'DAL')
symbols_list = ('MSFT', 'JPM', 'REGN', 'RCL', 'NCLH', 'LMT')
start_date = "2016-07-16"
end_date = "2021-07-16"
quote_channels = ('Close',)
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')
quotes, macros, _ = get_multiple_assets(symbols_list=symbols_list, start_date=start_date,
                                        end_date=end_date, quote_channels=quote_channels,
                                        adjust_prices=adjust_prices, cache_path=cache_path)

assets_data = [quotes[quote_channels[0]][:, i]
               for i in range(quotes[quote_channels[0]].shape[1])]
dates = quotes['Dates']
display_meta_paramets = (
    'five_years_div_yield',
    'book2value_ratio',
    'high_52w',
    'low_52w',
    'profit_margins',
    'trailing_price2earnings',
)
plot_assets_list(assets_symbols=symbols_list, assets_data=assets_data,
                 dates=dates, assets_meta_data=macros,
                 display_meta_paramets=display_meta_paramets
                 )
