from src.analysis.smoothing import Smoother
from src.analysis.forecasting import Forecaster

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


class TestForecaster:
    def test_forecast_exp(self, get_forecasting_data):
        forecast_horizon = 5
        x, y = get_forecasting_data

        # Instantiate a Smoother
        smoother = Smoother(method='exp', optimize=True)

        # Fit the smoother
        _ = smoother(y[:-forecast_horizon])

        # Instantiate a Forecaster
        forecaster = Forecaster(method='smoother', forecast_horizon=forecast_horizon,
                                smoother=smoother)

        # Generate forecast
        forecast = forecaster.forecast(y[:-forecast_horizon])

        # Check size
        assert len(forecast) == forecast_horizon

        # Forecast should be accurate roughly up to the scale of the i.i.d noise
        diff = np.mean(np.abs(forecast - x[-forecast_horizon:]))
        assert diff == pytest.approx(0, abs=1e-2)

    def test_forecast_holt_winters(self, get_forecasting_data):
        forecast_horizon = 5
        x, y = get_forecasting_data

        # Instantiate a Smoother
        smoother = Smoother(method='holt_winter', trend=None)

        # Fit the smoother
        _ = smoother(y[:-forecast_horizon])

        # Instantiate a Forecaster
        forecaster = Forecaster(method='smoother', forecast_horizon=forecast_horizon,
                                smoother=smoother)

        # Generate forecast
        forecast = forecaster.forecast(y[:-forecast_horizon])

        # Check size
        assert len(forecast) == forecast_horizon

        # Forecast should be accurate roughly up to the scale of the i.i.d noise
        diff = np.mean(np.abs(forecast - x[-forecast_horizon:]))
        assert diff == pytest.approx(0, abs=1e-2)

    def test_forecast_poly(self, get_forecasting_data_poly_deg2):
        forecast_horizon = 5
        x, y = get_forecasting_data_poly_deg2

        # Instantiate a Smoother
        smoother = Smoother(method='polyfit', poly_degree=2)

        # Instantiate a Forecaster
        forecaster = Forecaster(method='smoother', forecast_horizon=forecast_horizon,
                                smoother=smoother)

        # Generate forecast
        forecast = forecaster.forecast(y[:-forecast_horizon])

        # Check size
        assert len(forecast) == forecast_horizon

        # Forecast should be accurate roughly up to the scale of the i.i.d noise
        diff = np.mean(np.abs(forecast - x[-forecast_horizon:]))
        assert diff == pytest.approx(0, abs=3 * 1e-1)

    def test_arima_forecast(self, get_forecasting_data_arima):
        forecast_horizon = 5
        x, y = get_forecasting_data_arima
        fit_set = y[:-forecast_horizon]
        # Instantiate a Forecaster
        forecaster = Forecaster(method='arima', forecast_horizon=forecast_horizon,
                                arima_orders=(2, 1, 2))

        # Generate forecasts
        prediction_start_fit_set = 1
        prediction_end_fit_set = len(fit_set) - 1
        gt_forecast = forecaster.forecast(y[:-forecast_horizon],
                                          prediction_start=prediction_start_fit_set,
                                          prediction_end=prediction_end_fit_set)

        prediction_start_pred_set = len(fit_set)
        prediction_end_pred_set = len(fit_set) + forecast_horizon - 1
        pred_forecast = forecaster.forecast(y[:-forecast_horizon],
                                            prediction_start=prediction_start_pred_set,
                                            prediction_end=prediction_end_pred_set)

        # Check size
        assert len(gt_forecast) == len(fit_set) - 1
        assert len(pred_forecast) == forecast_horizon

        # Forecast should fit the training set up to the i.i.d noise scale
        diff = np.mean(np.abs(gt_forecast - fit_set[1:]))
        assert diff == pytest.approx(0, abs=1e-2)

        # Forecast should fit the test set up to the i.i.d noise scale
        diff = np.mean(np.abs(pred_forecast - x[-forecast_horizon:]))
        assert diff == pytest.approx(0, abs=1e-2)
