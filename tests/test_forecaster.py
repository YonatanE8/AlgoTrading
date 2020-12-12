from src.analysis.smoothing import Smoother
from tests.conftest import BasicTestingData
from src.analysis.forecasting import Forecaster

import numpy as np
import pytest


testing_data = BasicTestingData()


class TestForecaster:
    def test_forecast_exp(self):
        testing_data.get_forecasting_data()
        x, y = testing_data.x, testing_data.y
        forecast_horizon = 5

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

    def test_forecast_holt_winters(self):
        testing_data.get_forecasting_data()
        x, y = testing_data.x, testing_data.y
        forecast_horizon = 5

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

    def test_forecast_poly(self):
        testing_data.get_forecasting_data_poly_deg2()
        x, y = testing_data.x, testing_data.y
        forecast_horizon = 5

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

    def test_arima_forecast(self):
        testing_data.get_forecasting_data_arima()
        x, y = testing_data.x, testing_data.y
        forecast_horizon = 5
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