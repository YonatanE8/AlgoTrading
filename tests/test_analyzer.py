import pytest
import numpy as np


class TestAnalyzer:
    def test_compute_returns(self, get_analyzer):
        returns = get_analyzer.returns

        assert returns.shape[0] == get_analyzer.quotes.shape[0] - 1
        assert returns.shape[1] == get_analyzer.quotes.shape[1]
        assert returns.shape[1] == len(get_analyzer.symbols_list)
        assert np.max(returns) <= 0.2
        assert np.min(returns) >= -0.2

        synthetic_sig = np.arange(1, 101)
        synth_returns = get_analyzer._compute_returns(synthetic_sig)

        diff = np.sum(np.abs(synth_returns - (1 / synthetic_sig[:-1])))
        assert diff == pytest.approx(0, abs=1e-6)

    def test_analyze_sr(self, get_analyzer):
        sr = get_analyzer._analyze_sr()

        assert np.max(np.abs(sr)) <= 4

        diff = np.abs(sr[1:, :] - sr[:-1, :])
        diff = np.mean(diff, axis=0)
        for d in diff:
            assert d <= 0.3

    def test_analyze_returns_histogram(self, get_analyzer):
        values, counts = get_analyzer._analyze_returns_histogram()

        assert values.shape[0] == counts.shape[0] + 1
        assert values.shape[1] == counts.shape[1]
        assert values.shape[1] == len(get_analyzer.symbols_list)
        assert np.max(counts) < 1.
        assert np.min(counts) >= 0.
        assert np.max(np.abs(values)) < 0.25

    def test_compute_mean_annual_return(self, get_analyzer):
        mean_annual_returns = get_analyzer._compute_mean_annual_return()

        assert len(get_analyzer.symbols_list) == len(mean_annual_returns)
        assert np.max(np.abs(mean_annual_returns)) < 0.25

    def test_analyze_periodicty(self, get_analyzer):
        final_signal = get_analyzer._analyze_periodicty(get_analyzer.quotes)

        assert len(get_analyzer.symbols_list) == final_signal.shape[1]
        assert final_signal.shape[0] == get_analyzer.quotes.shape[0] - 1

        diff = (get_analyzer.quotes[1:, :] - final_signal) / get_analyzer.quotes[1:, :]
        max_diff = np.max(np.abs(diff), axis=0)
        mean_diff = np.mean(np.abs(diff), axis=0)

        for d in max_diff:
            assert d < 0.15

        for d in mean_diff:
            assert d < 0.03

    def test_returns_emerging_trend(self, get_analyzer):
        trend_mean, trend_std = get_analyzer._returns_emerging_trend(
            get_analyzer.quotes)

        assert len(trend_mean) == len(get_analyzer.symbols_list)
        assert len(trend_mean) == len(trend_std)
        assert np.max(np.abs(trend_mean)) <= 0.1
        assert np.max(np.abs(trend_std)) <= 0.1

    def test_compute_overall_period_return(self, get_analyzer):
        overall_returns = get_analyzer._compute_overall_period_return()

        assert len(overall_returns) == len(get_analyzer.symbols_list)
        assert np.max(np.abs(overall_returns)) <= 0.1

    def test_analyze(self, get_analyzer):
        analysis = get_analyzer.analyze()
        keys = [
            'sr',
            'mean',
            'recent_trend_mean',
            'recent_trend_std',
            'periodicity',
        ]

        for key in keys:
            assert key in analysis
            assert isinstance(analysis[key], np.ndarray)
