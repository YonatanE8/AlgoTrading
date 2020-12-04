import pytest


@pytest.fixture
def get_quote_params_yahoo():
    symbol = 'MSFT'
    start_date = "2010-01-01"
    end_date = "2011-01-01"
    data_source = 'yahoo'

    params = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'data_source': data_source,
    }

    return params


@pytest.fixture
def get_quote_params_pandas():
    symbol = 'MSFT'
    start_date = "2010-01-01"
    end_date = "2011-01-01"
    data_source = 'pandas'

    params = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'data_source': data_source,
    }

    return params




