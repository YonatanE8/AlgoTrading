def anaylze_assets(analysis_period_len: int = 1825, data_source: str = 'yahoo',
                   metric2_analyze: str = 'Close', mini_period_len: int = 66,
                   desired_peak_gain: float = 0.25, desired_mean_gain: float = 0.1,
                   start_date: str = None):
    """

    :param analysis_period_len:
    :param data_source:
    :param metric2_analyze:
    :param mini_period_len:
    :param desired_peak_gain:
    :param desired_mean_gain:
    :param start_date:
    :return:
    """

    sp500_tickers, tickers_data, tickers_meta_data, sp500_names = get_sp500_assets(
        analysis_period_len=analysis_period_len, data_source=data_source,
        metric2_analyze=metric2_analyze, start_date=start_date)

    # Analyze data
    n_trading_days = tickers_data.shape[0]
    n_assets = tickers_data.shape[1]

    print(f"Extracted {n_assets} assets")

    value_periods = [tickers_data[d:(d + mini_period_len), :]
                     for d in range(n_trading_days - mini_period_len)]
    avg_value_period = [np.mean(period, 0) for period in value_periods]
    avg_value_period = np.concatenate([np.expand_dims(period, 0)
                                       for period in avg_value_period], 0)
    max_value_period = np.max(avg_value_period, 0)
    mean_values = np.mean(avg_value_period, 0)
    current_values = np.mean(tickers_data[-mini_period_len:, :], 0)

    current2peak_ratio = current_values / max_value_period
    current2mean_ratio = current_values / mean_values

    peak_assets = np.where(current2peak_ratio <= desired_peak_gain)[0]
    mean_assets = np.where(current2mean_ratio <= desired_mean_gain)[0]

    interesting_assets_inds = np.intersect1d(peak_assets, mean_assets).tolist()
    interesting_assets_inds = [int(i) for i in interesting_assets_inds]
    interesting_assets_symbols = [sp500_tickers[i] for i in interesting_assets_inds]
    interesting_assets_meta_data = [tickers_meta_data[i]
                                    for i in interesting_assets_inds]
    interesting_assets_data = [tickers_data[:, i] for i in interesting_assets_inds]
    names = [sp500_names[i] for i in interesting_assets_inds]

    plot_assets_list(assets_symbols=interesting_assets_symbols,
                     assets_data=interesting_assets_data,
                     assets_meta_data=interesting_assets_meta_data,
                     names=names)