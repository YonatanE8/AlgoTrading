from FinancialAnalysis import PROJECT_ROOT
from FinancialAnalysis.analysis.smoothing import Smoother
from FinancialAnalysis.analysis.forecasting import Forecaster
from FinancialAnalysis.stocks_io.data_queries import get_asset_data
from FinancialAnalysis.visualizations.plot_assets import plot_forecasts

import os
import plotly.graph_objs as go

# Define data parameters
# symbols_list = ('MSFT', 'DIS', 'JPM', 'C', 'DAL')
symbols_list = ('MSFT', )
start_date = "2021-01-01"
end_date = "2021-03-10"
quote_channels = ('Close',)
adjust_prices = True
cache_path = os.path.join(PROJECT_ROOT, 'data')

# Define Smoother
# smoother = Smoother(method='avg', length=5)
# smoother = Smoother(method='exp', alpha=0.6, optimize=False)
# smoother = Smoother(method='exp', optimize=True)
smoother = Smoother(method='holt_winter', trend=None)
# smoother = Smoother(method='polyfit', poly_degree=2)

# Define Forecaster
method = 'arima'
# method = 'smoother'
forecast_horizon = 2
arima_orders = (2, 0, 1)
remove_mean = False
arima_prediction_type = 'levels'
forecaster = Forecaster(
    method=method,
    forecast_horizon=forecast_horizon,
    smoother=smoother,
    arima_orders=arima_orders,
    arima_prediction_type=arima_prediction_type,
    remove_mean=remove_mean,
)

# Plotting parameters
display_meta_paramets = (
    'five_years_div_yield',
    'book2value_ratio',
    'high_52w',
    'low_52w',
    'profit_margins',
    'trailing_price2earnings',
)
if __name__ == '__main__':
    for symbol in symbols_list:
        # Get data
        quotes, macros = get_asset_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            quote_channels=quote_channels,
            adjust_prices=adjust_prices,
            cache_path=cache_path,
        )
        asset_data = quotes[quote_channels[0]]
        dates = quotes['Dates']
        window_len = 20

        # Plot forecast
        figure = go.Figure()
        plot_forecasts(
            time_series=asset_data,
            window_len=window_len,
            smoother=smoother,
            forecaster=forecaster,
            asset_symbol=symbol,
            dates=dates,
            figure=figure,
        )
