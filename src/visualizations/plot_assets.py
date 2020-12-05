from os import linesep
from datetime import datetime

import matplotlib

matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_assets_list(assets_symbols: tuple, assets_data: list, dates: list,
                     assets_meta_data: list) -> None:
    """

    :param assets_symbols: (list) A list of strings, denoting the listed symbols
    of the assets to be plotted
    :param assets_data: (list) A list of NumPy arrays, denoting the stocks to plot
    :param dates: (list) A list of strings, denoting the dates for which
    the quotes are given
    :param assets_meta_data: (list) A list of dicts containing the macro data for
    each asset to be plotted

    :return: None
    """

    n_assets = len(assets_data)
    colors = sns.color_palette('husl', n_assets)
    fig, ax = plt.subplots(figsize=[16, 9])
    names = []
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    with sns.axes_style("darkgrid"):
        for i, asset in enumerate(assets_data):
            meta_info = assets_meta_data[i]
            info = f'{linesep}'.join([f'{key}: {meta_info[key]}' for key in meta_info])
            info = (f'{linesep}{"*" * 30}{linesep}' + info +
                    f'{linesep}{"*" * 30}{linesep}')
            print(info)

            ax.plot(dates, asset, c=colors[i])
            ax.set_ylabel('Price [USD]')
            ax.set_xlabel('Date')
            names.append(f"{assets_symbols[i]}: {meta_info['name']}")

        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        fig.autofmt_xdate()
        plt.legend(names)
        plt.show()
