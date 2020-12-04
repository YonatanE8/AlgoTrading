import pytest


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





