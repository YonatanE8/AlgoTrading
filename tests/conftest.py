from src import PROJECT_ROOT
from src.analysis.analyzing import Analyzer

import os
import pytest
import numpy as np


np.random.seed(42)


@pytest.fixture
def get_quote_params():
    symbol = 'MSFT'
    start_date = "2010-01-01"
    end_date = "2011-01-01"
    quote_channels = ('Close', 'Open', 'High', 'Low', 'Volume')
    adjust_prices = True
    params = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'quote_channels': quote_channels,
        'adjust_prices': adjust_prices,
    }

    return params


@pytest.fixture
def get_multple_quotes_params():
    symbols_list = ('MSFT', 'GOOGL', 'AAPL')
    start_date = "2010-01-01"
    end_date = "2011-01-01"
    quote_channels = ('Close', 'Open', 'High', 'Low', 'Volume')
    adjust_prices = True
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'tmp_test')
    os.makedirs(cache_path, exist_ok=True)

    params = {
        'symbols_list': symbols_list,
        'start_date': start_date,
        'end_date': end_date,
        'quote_channels': quote_channels,
        'adjust_prices': adjust_prices,
        'cache_path': cache_path,
    }

    return params


@pytest.fixture
def get_exp_smooth_params():
    method = 'exp'
    alpha = 0.6
    optimize = True

    params = {
        'method': method,
        'alpha': alpha,
        'optimize': optimize,
    }

    return params


@pytest.fixture
def get_holt_winters_smoothing_params():
    method = 'holt_winter'
    trend = 'additive'

    params = {
        'method': method,
        'trend': trend,
    }

    return params


@pytest.fixture
def get_running_window_smoothing_params():
    method = 'avg'
    length = 10

    params = {
        'method': method,
        'length': length,
    }

    return params


@pytest.fixture
def get_fit_polyfit_params():
    method = 'polyfit'
    poly_degree = 3

    params = {
        'method': method,
        'poly_degree': poly_degree,
    }

    return params


class BasicTestingData:
    def get_linear(self):
        self.x = np.linspace(start=-1.0, stop=1.0, num=150000)
        self.y = self.x + np.random.normal(loc=0, scale=0.1, size=(len(self.x, )))

    def get_poly_deg2(self):
        x = np.linspace(start=-1, stop=1, num=10000)
        self.x = (-2 * x) + np.power(x, 2)
        self.y = self.x + np.random.normal(loc=0, scale=0.1, size=(len(self.x, )))

    def get_forecasting_data(self):
        noise_scale = 0.1
        window_length = 100000
        trend = np.linspace(start=-2, stop=2, num=window_length)
        scale = np.random.uniform(low=-0.5, high=0.5, size=(1, ))
        offset = np.random.uniform(low=-2, high=2, size=(1, ))
        noise = np.random.normal(loc=0, scale=noise_scale, size=(window_length, ))

        self.x = offset + scale * trend
        self.y = self.x + noise

    def get_forecasting_data_poly_deg2(self):
        noise_scale = 0.1
        window_length = 100000
        trend = np.linspace(start=-2, stop=2, num=window_length)
        trend = (1.5 * trend) + (-2 * np.power(trend, 2))
        offset = np.random.uniform(low=-2, high=2, size=(1, ))
        noise = np.random.normal(loc=0, scale=noise_scale, size=(window_length, ))

        self.x = offset + trend
        self.y = self.x + noise

    def get_forecasting_data_arima(self):
        noise_scale = 0.01
        window_length = 100
        trend = np.linspace(start=-2, stop=2, num=window_length)
        scale = np.random.uniform(low=-0.5, high=0.5, size=(1, ))
        offset = np.random.uniform(low=-2, high=2, size=(1, ))
        noise = np.random.normal(loc=0, scale=noise_scale, size=(window_length, ))

        self.x = offset + scale * trend
        self.y = self.x + noise


@pytest.fixture
def get_analyzer():
    symbols_list = ("MSFT", "AAPL", "JPM", "C", "DIS")
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    quote_channel = 'Close'
    adjust_prices = True
    risk_free_asset_symbol = '^IRX'
    bins = 10
    spectral_energy_threshold = 0.05
    trend_period_length = 10
    cache_path = os.path.join(PROJECT_ROOT, 'tests', 'test_cache')
    os.makedirs(cache_path, exist_ok=True)

    analyzer = Analyzer(symbols_list=symbols_list, start_date=start_date,
                        end_date=end_date, quote_channel=quote_channel,
                        adjust_prices=adjust_prices,
                        risk_free_asset_symbol=risk_free_asset_symbol,
                        bins=bins, spectral_energy_threshold=spectral_energy_threshold,
                        trend_period_length=trend_period_length,
                        cache_path=cache_path)

    return analyzer


