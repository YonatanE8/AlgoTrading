from src.analysis.scanning import Scanner

import pytest


class TestScanner:
    def test_set_criterions(self, get_scanner):
        macro_criterion = {
            'some_false_criterion': (0, 1)
        }

        quote_criterion = {
            'some_false_criterion': (0, 1)
        }

        with pytest.raises(AssertionError):
            get_scanner.set_macro_criterions(macro_criterion)

        with pytest.raises(AssertionError):
            get_scanner.set_quote_criterions(quote_criterion)

    


