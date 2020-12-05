from src import PROJECT_ROOT
from src.io.data_queries import get_multiple_assets
from src.analysis.smoothing import Smoother
from src.visualizations.plot_assets import plot_smooth_assets_list

import os

# symbols_list = ('MSFT', 'DIS', 'JPM', 'C', 'DAL')
symbols_list = ('JPM', 'C', 'DIS', 'DAL')
start_date = "2015-12-03"
end_date = "2020-12-03"
quote_channels = ('Close',)
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')
quotes, macros = get_multiple_assets(symbols_list=symbols_list, start_date=start_date,
                                     end_date=end_date, quote_channels=quote_channels,
                                     adjust_prices=adjust_prices, cache_path=cache_path)

assets_data = [quotes[quote_channels[0]][:, i]
               for i in range(quotes[quote_channels[0]].shape[1])]
dates = quotes['Dates']
smoothers = [
    # Smoother(method='avg', length=15),
    # Smoother(method='exp', alpha=0.6, optimize=False),
    # Smoother(method='exp', optimize=True),
    # Smoother(method='holt_winter', trend=None),
    Smoother(method='polyfit', poly_degree=15),
]
linewidth = 1.
markersize = 2.5
alpha = 0.4
plot_smooth_assets_list(assets_symbols=symbols_list, assets_data=assets_data,
                        smoothers=smoothers, dates=dates, assets_meta_data=macros,
                        linewidth=linewidth, markersize=markersize, alpha=alpha)