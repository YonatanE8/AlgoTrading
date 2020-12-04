from src.io.data_queries import (
    get_sp500_symbols_wiki,
    get_nasdaq_listed_symbols,
    get_quotes,
)

import os
import pytest


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

    def test_get_quotes(self, get_quote_params_yahoo, get_quote_params_pandas):
        quote = get_quotes(**get_quote_params_yahoo)

        quote = get_quotes(**get_quote_params_pandas)


