from src import PROJECT_ROOT

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
def get_linear():
    x = np.linspace(start=-1.0, stop=1.0, num=150000)
    y = x + np.random.normal(loc=0, scale=0.1, size=(len(x, )))

    return x, y


@pytest.fixture
def get_poly_deg2():
    x = np.linspace(start=-1, stop=1, num=10000)
    x = (-2 * x) + np.power(x, 2)
    y = x + np.random.normal(loc=0, scale=0.1, size=(len(x, )))

    return x, y


@pytest.fixture
def get_fit_polyfit_params():
    method = 'polyfit'
    poly_degree = 3

    params = {
        'method': method,
        'poly_degree': poly_degree,
    }

    return params


@pytest.fixture
def get_forecasting_data():
    noise_scale = 0.1
    window_length = 100000
    trend = np.linspace(start=-2, stop=2, num=window_length)
    scale = np.random.uniform(low=-0.5, high=0.5, size=(1, ))
    offset = np.random.uniform(low=-2, high=2, size=(1, ))
    noise = np.random.normal(loc=0, scale=noise_scale, size=(window_length, ))

    x = offset + scale * trend
    y = x + noise

    return x, y
