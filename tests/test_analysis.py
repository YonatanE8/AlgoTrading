from src.io.data_queries import get_asset_data
from src.analysis.smoothing import Smoother

import pytest
import numpy as np


class TestSmoother:
    def test_exponential_smoothing(self, get_linear, get_exp_smooth_params):
        x, y = get_linear
        smoother = Smoother(**get_exp_smooth_params)
        smoothed = smoother(y)

        # Test shapes
        assert smoothed.shape == x.shape

        # The average error should be smaller then the scale of the i.i.d noise
        diff = np.abs(x - smoothed)
        assert np.mean(diff) == pytest.approx(0, abs=1e-2)

    def test_holt_winters_smoothing(self, get_poly_deg2,
                                    get_holt_winters_smoothing_params):
        x, y = get_poly_deg2
        smoother = Smoother(**get_holt_winters_smoothing_params)
        smoothed = smoother(y)

        # Test shapes
        assert smoothed.shape == x.shape

        # The average error should be smaller then the scale of the i.i.d noise
        diff = np.abs(x - smoothed)
        assert np.mean(diff) == pytest.approx(0, abs=1e-2)

    def test_running_window_smoothing(self, get_linear,
                                      get_running_window_smoothing_params):
        x, y = get_linear
        smoother = Smoother(**get_running_window_smoothing_params)
        smoothed = smoother(y)

        # Test shapes
        assert smoothed.shape == x.shape

        # The average error should be on the same scale as the i.i.d noise on the
        # valid part of the convolution
        diff = np.abs(x[smoother.length:-smoother.length] -
                      smoothed[smoother.length:-smoother.length])
        assert np.mean(diff) == pytest.approx(0, abs=3e-2)

    def test_fit_polyfit(self, get_fit_polyfit_params):
        time_series = np.arange(1000)
        time_series = (-2 * time_series) * np.power(time_series, 2)
        smoother = Smoother(**get_fit_polyfit_params)
        smoothed = smoother(time_series)

        # Test shapes
        assert smoothed.shape == time_series.shape

        # Polyfit should fit the data perfectly
        diff = np.abs(time_series - smoothed)
        assert np.sum(diff) == pytest.approx(0, abs=1e-3)

