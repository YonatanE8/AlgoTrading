from src.io.data_queries import get_asset_data
from src.analysis.statistical_analysis import Smoother

import numpy as np


class TestAnalysis:
    def test_exponential_smoothing(self, get_quote_params, get_exp_smooth_params):
        quotes, macros = get_asset_data(**get_quote_params)

        time_series = quotes['Close']
        smoother = Smoother(**get_exp_smooth_params)
        smoothed = smoother(time_series)

        # Test shapes
        assert smoothed.shape == time_series.shape

        # The smoothed series should have a strictly lower std,
        # maximum day-to-day difference and daily change then the original series.
        diff_ts = np.abs(np.diff(time_series))
        diff_sm = np.abs(np.diff(smoothed))

        assert np.mean(diff_ts) > np.mean(diff_sm)
        assert np.max(diff_ts) > np.max(diff_sm)
        assert np.std(time_series) > np.std(smoothed)

    def test_holt_winters_smoothing(self, get_quote_params,
                                    get_holt_winters_smoothing_params):
        quotes, macros = get_asset_data(**get_quote_params)

        time_series = quotes['Close']
        smoother = Smoother(**get_holt_winters_smoothing_params)
        smoothed = smoother(time_series)

        # Test shapes
        assert smoothed.shape == time_series.shape

        # The smoothed series should have lower or equal maximum day-to-day difference
        # and daily change then the original series.
        diff_ts = np.abs(np.diff(time_series))
        diff_sm = np.abs(np.diff(smoothed))

        assert np.mean(diff_ts) >= np.mean(diff_sm)
        assert np.max(diff_ts) >= np.max(diff_sm)

    def test_running_window_smoothing(self, get_quote_params,
                                      get_running_window_smoothing_params):
        quotes, macros = get_asset_data(**get_quote_params)

        time_series = quotes['Close']
        smoother = Smoother(**get_running_window_smoothing_params)
        smoothed = smoother(time_series)

        # Test shapes
        assert smoothed.shape == time_series.shape

        # The smoothed series should have a strictly lower daily change then the
        # original series.
        diff_ts = np.abs(np.diff(time_series))
        diff_sm = np.abs(np.diff(smoothed))

        assert np.mean(diff_ts) > np.mean(diff_sm)



