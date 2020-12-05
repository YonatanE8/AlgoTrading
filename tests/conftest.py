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
    optimize = False

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
