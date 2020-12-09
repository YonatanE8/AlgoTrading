from os import linesep
from datetime import datetime
from src.analysis.smoothing import Smoother
from src.analysis.forecasting import Forecaster
import matplotlib

matplotlib.use('TkAgg')

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_assets_list(assets_symbols: tuple, assets_data: list, dates: list,
                     assets_meta_data: list, save_plot_path: str = None,
                     save_report_path: str = None) -> None:
    """
    A method for plotting a list of tradable equities

    :param assets_symbols: (list) A list of strings, denoting the listed symbols
    of the assets to be plotted
    :param assets_data: (list) A list of NumPy arrays, denoting the stocks to plot
    :param dates: (list) A list of strings, denoting the dates for which
    the quotes are given
    :param assets_meta_data: (list) A list of dicts containing the macro data for
    each asset to be plotted
    :param save_plot_path: (str) Path to a directory in which to save the figure,
    doesn't save if None, default is None.
    :param save_report_path: (str) Path to a directory in which to save the
    per-asset report, doesn't save if None, default is None.

    :return: None
    """

    n_assets = len(assets_data)
    colors = sns.color_palette('husl', n_assets)
    fig, ax = plt.subplots(figsize=[16, 9])
    names = []
    infos = []
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    with sns.axes_style("darkgrid"):
        for i, asset in enumerate(assets_data):
            meta_info = assets_meta_data[i]
            info = f'{linesep}'.join([f'{key}: {meta_info[key]}' for key in meta_info])
            info = (f'{linesep}{"*" * 30}{linesep}' + info +
                    f'{linesep}{"*" * 30}{linesep}')
            print(info)
            infos.append(info)

            ax.plot(dates[-len(asset):], asset, c=colors[i])
            names.append(f"{assets_symbols[i]}: {meta_info['name']}")

        ax.set_ylabel('Price [USD]')
        ax.set_xlabel('Date')
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        plt.legend(names)

        if save_plot_path is not None:
            if not save_plot_path.endswith('.pdf'):
                save_plot_path += '.pdf'

            plt.savefig(fname=os.path.join(save_plot_path),
                        dpi=300, orientation='landscape', format='pdf')

        if save_report_path is not None:
            if not save_report_path.endswith('.txt'):
                save_report_path += '.txt'

            start_date = dates[0]
            end_date = dates[-1]
            with open(save_report_path, 'w') as report_file:
                report_file.write(f"Asset report from {start_date} - {end_date}\n\n")
                [report_file.write(info) for info in infos]

        plt.show()


def plot_smooth_assets_list(
        assets_symbols: tuple, assets_data: list, dates: list, assets_meta_data: list,
        smoothers: (Smoother, ...), save_plot_path: str = None,
        save_report_path: str = None, linewidth: float = 1.0,
        markersize: float = 1.0, alpha: float = 1.0) -> None:
    """
    A method for plotting a list of tradable equities after smoothening.

    :param assets_symbols: (list) A list of strings, denoting the listed symbols
    of the assets to be plotted
    :param assets_data: (list) A list of NumPy arrays, denoting the stocks to plot
    :param dates: (list) A list of strings, denoting the dates for which
    the quotes are given
    :param assets_meta_data: (list) A list of dicts containing the macro data for
    each asset to be plotted
    :param smoothers: (list) A list of Smoother class instances. Each smoother will be
    applied to all assets and plotted together.
    :param save_plot_path: (str) Path to a directory in which to save the figure,
    doesn't save if None, default is None.
    :param save_report_path: (str) Path to a directory in which to save the
    per-asset report, doesn't save if None, default is None.
    :param linewidth: (float) Width of the plotted line, default is 1.0.
    :param markersize: (float) Size of the plotted markers, default is 1.0.
    :param alpha: (float) Shade intensity for the raw data plot, default is 1.0.

    :return: None
    """

    n_assets = len(assets_data)
    colors = sns.color_palette('husl', n_assets)
    fig, ax = plt.subplots(figsize=[16, 9])
    names = []
    infos = []
    linestyles = ['--', '-.', ':']
    markers = ['o', 'D', '*', 'v', '8', 's', 'p']
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    with sns.axes_style("darkgrid"):
        for i, asset in enumerate(assets_data):
            # Extract the macro info
            meta_info = assets_meta_data[i]
            info = f'{linesep}'.join([f'{key}: {meta_info[key]}'
                                      for key in meta_info])
            info = (f'{linesep}{"*" * 30}{linesep}' + info +
                    f'{linesep}{"*" * 30}{linesep}')
            print(info)
            infos.append(info)

            # Plot the un-smoothed data
            ax.plot(dates[-len(asset):], asset, c=colors[i], linestyle='-', alpha=alpha)
            names.append(f"{assets_symbols[i]}: {meta_info['name']}")

            # Plot all smoothers
            for s, smoother in enumerate(smoothers):
                smoothed_asset = smoother(asset)
                ax.plot(dates[-len(smoothed_asset):], smoothed_asset,
                        c=colors[i], linestyle=linestyles[(s % len(linestyles))],
                        marker=markers[(s % len(markers))], linewidth=linewidth,
                        markersize=markersize)
                names.append(
                    f"{assets_symbols[i]}: {meta_info['name']} - {smoother.description}"
                )

        ax.set_ylabel('Price [USD]')
        ax.set_xlabel('Date')
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        plt.legend(names)

        if save_plot_path is not None:
            if not save_plot_path.endswith('.pdf'):
                save_plot_path += '.pdf'

            plt.savefig(fname=os.path.join(save_plot_path),
                        dpi=300, orientation='landscape', format='pdf')

        if save_report_path is not None:
            if not save_report_path.endswith('.txt'):
                save_report_path += '.txt'

            start_date = dates[0]
            end_date = dates[-1]
            with open(save_report_path, 'w') as report_file:
                report_file.write(f"Asset report from {start_date} - {end_date}\n\n")
                [report_file.write(info) for info in infos]

        plt.show()


def plot_forecasts(periods: (np.ndarray, ...), smoother: Smoother,
                   forecaster: Forecaster, assets_symbol: str,
                   dates: list, asset_meta_data: dict,
                   save_plot_path: str = None,
                   save_report_path: str = None, linewidth: float = 1.0,
                   markersize: float = 1.0, alpha: float = 1.0) -> None:
    """
    A method for plotting forecasts for a list of tradable equity after smoothening.

    :param periods: (Tuple) tuple of NumPy arrays, each denoting a period to perform
    forecasting on, should be of length of at least 2 periods.
    :param smoother: (Smoother) A smoother object to use on each period
    :param forecaster: (Forecaster) A forecaster object to calculate the forecast over
    the period
    :param assets_symbol: (str) A string denoting the symbol
    of the asset to be plotted
    :param dates: (list) A list of strings, denoting the dates for which
    the quotes are given
    :param asset_meta_data: (dict) A dict containing the macro data for
    the asset to be plotted
    :param save_plot_path: (str) Path to a directory in which to save the figure,
    doesn't save if None, default is None.
    :param save_report_path: (str) Path to a directory in which to save the
    per-asset report, doesn't save if None, default is None.
    :param linewidth: (float) Width of the plotted line, default is 1.0.
    :param markersize: (float) Size of the plotted markers, default is 1.0.
    :param alpha: (float) Shade intensity for the raw data plot, default is 1.0.

    :return: None
    """

    smoothed_list = []
    forecasts = []
    for i, period in enumerate(periods[1:]):
        period = np.concatenate(periods[:i + 1])
        smoothed = smoother(period)

        forecaster.smoother = smoother
        forecaster.arima_model = None

        forecast = forecaster(smoothed)

        smoothed_list.append(smoothed)
        forecasts.append(forecast)

    names = []
    linestyles = ['--', '-.', ':']
    markers = ['o', 'D', '*', 'v', '8', 's', 'p']
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    fig, ax = plt.subplots(figsize=[16, 9])
    with sns.axes_style("darkgrid"):
        # Extract the macro info
        info = f'{linesep}'.join([f'{key}: {asset_meta_data[key]}'
                                  for key in asset_meta_data])
        info = (f'{linesep}{"*" * 30}{linesep}' + info +
                f'{linesep}{"*" * 30}{linesep}')
        print(info)

        # Plot the smoothed data
        ax.plot(dates[-len(smoothed[-1]):], smoothed[-1], linestyle='-', alpha=alpha)
        names.append(f"{assets_symbol}: {asset_meta_data['name']}")

        # Plot all forecasts
        for f, forecast in enumerate(forecasts):
            ax.plot(dates[-len(forecast):], forecast,
                    linestyle=linestyles[(f % len(linestyles))],
                    marker=markers[(f % len(markers))], linewidth=linewidth,
                    markersize=markersize)
            names.append(
                f"Forecast Period {f}"
            )

        ax.set_ylabel('Price [USD]')
        ax.set_xlabel('Date')
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        plt.legend(names)

        if save_plot_path is not None:
            if not save_plot_path.endswith('.pdf'):
                save_plot_path += '.pdf'

            plt.savefig(fname=os.path.join(save_plot_path),
                        dpi=300, orientation='landscape', format='pdf')

        if save_report_path is not None:
            if not save_report_path.endswith('.txt'):
                save_report_path += '.txt'

            start_date = dates[0]
            end_date = dates[-1]
            with open(save_report_path, 'w') as report_file:
                report_file.write(f"Asset report from {start_date} - {end_date}\n\n")
                [report_file.write(info) for info in infos]

        plt.show()
