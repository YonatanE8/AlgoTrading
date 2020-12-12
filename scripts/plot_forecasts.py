from src import PROJECT_ROOT
from src.stocks_io.data_queries import get_asset_data
from src.analysis.smoothing import Smoother
from src.analysis.forecasting import Forecaster
from src.visualizations.plot_assets import plot_forecasts

import os

# Define data parameters
symbols_list = ('MSFT', 'DIS', 'JPM', 'C', 'DAL')
start_date = "2015-12-03"
end_date = "2020-12-03"
quote_channels = ('Close',)
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')

# Define Smoother
# smoother = Smoother(method='avg', length=15)
# smoother = Smoother(method='exp', alpha=0.6, optimize=False)
# smoother = Smoother(method='exp', optimize=True)
# smoother = Smoother(method='holt_winter', trend=None)
smoother = Smoother(method='polyfit', poly_degree=4)

# Define Forecaster
# method = 'arima'
method = 'smoother'
forecast_horizon = 22
period_length = 66

arima_orders = (5, 2, 1)
arima_prediction_type = 'levels'
forecaster = Forecaster(method=method, forecast_horizon=forecast_horizon,
                        smoother=smoother, arima_orders=arima_orders,
                        arima_prediction_type=arima_prediction_type)

# Plotting parameters
linewidth = 1.
markersize = 2.5
alpha = 0.4
save_plot_path = None
save_report_path = None

if __name__ == '__main__':
    for symbol in symbols_list:
        # Get data
        quotes, macros = get_asset_data(symbol=symbol, start_date=start_date,
                                        end_date=end_date,
                                        quote_channels=quote_channels,
                                        adjust_prices=adjust_prices,
                                        cache_path=cache_path)
        assets_data = quotes[quote_channels[0]]
        n_periods = len(assets_data) // period_length
        periods = [assets_data[i * period_length:((i + 1) * period_length)]
                   for i in range(n_periods)]
        dates = quotes['Dates']

        if save_plot_path is not None:
            save_plot_path_symbol = os.path.join(save_plot_path, (symbol + '.png'))

        else:
            save_plot_path_symbol = None

        if save_report_path is not None:
            save_report_path_symbol = os.path.join(save_report_path, (symbol + '.png'))

        else:
            save_report_path_symbol = None

        # Plot forecast
        plot_forecasts(periods=periods, smoother=smoother, forecaster=forecaster,
                       asset_symbol=symbol, dates=dates, asset_meta_data=macros,
                       save_plot_path=save_plot_path_symbol,
                       save_report_path=save_report_path_symbol,
                       linewidth=linewidth, markersize=markersize, alpha=alpha)
