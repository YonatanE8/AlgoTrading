from datetime import datetime
from src.utils.hashing import dict_hash
from src.io.data_queries import (
    get_sp500_symbols_wiki,
    get_nasdaq_listed_symbols,
    get_asset_data,
    get_multiple_assets,
)

import os
import shutil
import numpy as np


class TestDataQueries:
    def test_get_nasdaq_listed_symbols(self):
        symbols, names = get_nasdaq_listed_symbols()

        assert isinstance(symbols, list)
        assert isinstance(names, list)
        for i, sym in enumerate(symbols):
            assert isinstance(sym, str)
            assert sym.isupper()
            assert len(sym) <= 6  # No more then six characters per symbol

            assert isinstance(names[i], str)
            assert os.sep not in names[i]
            assert names[i][-1] != ' '

    def test_get_sp500_symbols_wiki(self):
        symbols, names = get_sp500_symbols_wiki()

        assert isinstance(symbols, list)
        assert isinstance(names, list)
        for i, sym in enumerate(symbols):
            assert isinstance(sym, str)
            assert sym.isupper()
            assert len(sym) <= 6  # No more then six characters per symbol

            assert isinstance(names[i], str)
            assert os.sep not in names[i]
            assert names[i][-1] != ' '

    def test_get_quotes(self, get_quote_params):
        quotes, macros = get_asset_data(**get_quote_params)

        assert 'Dates' in quotes
        assert len(quotes['Dates']) == 253
        assert quotes[get_quote_params['quote_channels'][0]].shape == (253, )
        for quote in get_quote_params['quote_channels']:
            assert quote in quotes
            assert (quotes[quote].shape ==
                    quotes[get_quote_params['quote_channels'][0]].shape)

        macro_numeric_keys = [
            'beta', 'dividend_rate', 'five_years_div_yield',
            'trailing_price2earnings', 'trailing_price2sales', 'book2value_ratio',
            'profit_margins', 'high_52w', 'low_52w', 'change_52w',
            'earnings_quarterly_growth'
        ]
        macro_str_keys = [
            'name', 'sector'
        ]
        macro_datetime_keys = [
            'last_dividend_date',
        ]

        for key in macro_numeric_keys:
            assert key in macros
            assert isinstance(macros[key], float)

        for key in macro_str_keys:
            assert key in macros
            assert isinstance(macros[key], str)

        for key in macro_datetime_keys:
            assert key in macros
            assert isinstance(macros[key], datetime)

    def test_get_multiple_assets(self, get_multple_quotes_params):
        np.random.seed(42)
        quotes, macros = get_multiple_assets(**get_multple_quotes_params)

        assert len(macros) == len(get_multple_quotes_params['symbols_list'])
        assert len(quotes) == len(get_multple_quotes_params['quote_channels']) + 1
        assert len(quotes['Dates']) == 253
        assert (quotes[get_multple_quotes_params['quote_channels'][0]].shape ==
                (253, len(get_multple_quotes_params['symbols_list'])))
        for quote in get_multple_quotes_params['quote_channels']:
            assert quote in quotes
            assert (quotes[quote].shape ==
                    quotes[get_multple_quotes_params['quote_channels'][0]].shape)

        hash_signature = dict_hash({key: get_multple_quotes_params[key]
                                    for key in get_multple_quotes_params
                                    if key != 'cache_path'})
        data_file = os.path.join(get_multple_quotes_params['cache_path'],
                                 (hash_signature + '.pkl'))
        assert os.path.isfile(data_file)

        # Cached data must be identical to the loaded data
        quotes_2, macros_2 = get_multiple_assets(**get_multple_quotes_params)
        assert len(quotes) == len(quotes_2)
        assert len(macros) == len(macros_2)

        for quote in get_multple_quotes_params['quote_channels']:
            assert np.sum(np.abs(quotes[quote] - quotes_2[quote])) == 0

        for i in range(len(macros)):
            for key in macros[i]:
                if macros[i][key] is not None:
                    assert macros[i][key] == macros_2[i][key]

                else:
                    assert macros_2[i][key] is None

        shutil.rmtree(get_multple_quotes_params['cache_path'])

