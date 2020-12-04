import matplotlib

matplotlib.use('TkAgg')



def plot_assets_list(assets_symbols: list, assets_data: list, assets_meta_data: list,
                     names: list):
    """

    :param assets_symbols:
    :param assets_data:
    :param assets_meta_data:
    :param names:
    :return:
    """

    n_assets = len(assets_symbols)
    for i in range(n_assets):
        plt.figure(figsize=[16, 9])

        meta_info = assets_meta_data[i]
        info = ', '.join([f'{key} = {meta_info[key]}' for key in meta_info])
        name = names[i] + ' - ' + assets_symbols[i] + ': ' + info

        plt.plot(assets_data[i])
        plt.title(name)
        plt.show()

def plot_sector(sector: str, start: datetime, end: datetime):
    """

    :param sector:
    :param start:
    :param end:
    :return:
    """

    if sector == 'tech':
        # Technology
        microsoft = get_quotes(symbol='MSFT', start_date=start, end_date=end,
                               data_source='yahoo')
        google = get_quotes(symbol='goog', start_date=start, end_date=end,
                            data_source='yahoo')
        tesla = get_quotes(symbol='TSLA', start_date=start, end_date=end,
                           data_source='yahoo')
        amazon = get_quotes(symbol='AMZN', start_date=start, end_date=end,
                            data_source='yahoo')
        apple = get_quotes(symbol='AAPL', start_date=start, end_date=end,
                           data_source='yahoo')
        netflix = get_quotes(symbol='NFLX', start_date=start, end_date=end,
                             data_source='yahoo')

        data_readers = [microsoft, google, tesla, amazon, apple, netflix]
        names = ['Microsoft', 'Alphabet', 'Tesla', 'Amazon', 'Apple', 'Netflix']

    elif sector == 'finance':
        # Finance
        goldman_sachs = get_quotes(symbol='GS', start_date=start, end_date=end,
                                   data_source='yahoo')
        jp_morgan = get_quotes(symbol='JPM', start_date=start, end_date=end,
                               data_source='yahoo')
        mastercard = get_quotes(symbol='MA', start_date=start, end_date=end,
                                data_source='yahoo')
        american_express = get_quotes(symbol='AXP', start_date=start, end_date=end,
                                      data_source='yahoo')
        visa = get_quotes(symbol='V', start_date=start, end_date=end,
                          data_source='yahoo')
        paypal = get_quotes(symbol='PYPL', start_date=start, end_date=end,
                            data_source='yahoo')
        data_readers = [goldman_sachs, jp_morgan, mastercard, american_express, visa,
                        paypal]
        names = ['Goldman Sachs', 'JP Morgan', 'Mastercard', 'American Express', 'Visa',
                 'PayPal']

    elif sector == 'food':
        # Food & Beverages
        mcdonalds = get_quotes(symbol='MCD', start_date=start, end_date=end, data_source='yahoo')
        coca_cola = get_quotes(symbol='KO', start_date=start, end_date=end, data_source='yahoo')
        pepsi_co = get_quotes(symbol='PEP', start_date=start, end_date=end, data_source='yahoo')
        hormel_foods = get_quotes(symbol='HRL', start_date=start, end_date=end, data_source='yahoo')
        data_readers = [mcdonalds, coca_cola, pepsi_co, hormel_foods]
        names = ["McDonald's", 'Coca Cola', 'Pepsi Co', 'Hormel Foods']

    elif sector == 'retail':
        # Retail
        procter_and_gamble = get_quotes(symbol='PG', start_date=start, end_date=end,
                                        data_source='yahoo')
        kimberly_clark = get_quotes(symbol='KMB', start_date=start, end_date=end,
                                    data_source='yahoo')
        wal_mart = get_quotes(symbol='WMT', start_date=start, end_date=end,
                              data_source='yahoo')
        walgreens = get_quotes(symbol='WBA', start_date=start, end_date=end,
                               data_source='yahoo')
        colgate_palmolive = get_quotes(symbol='CL', start_date=start, end_date=end,
                                       data_source='yahoo')
        data_readers = [procter_and_gamble, kimberly_clark, wal_mart, walgreens,
                        colgate_palmolive]
        names = ['Procter & Gamble', 'Kimberly Clark', 'WalMart', 'Walgreens',
                 'Colgate Palmolive']

    elif sector == 'lux':
        # Luxury
        brown_forman = get_quotes(symbol='BF.B', start_date=start, end_date=end,
                                  data_source='yahoo')
        lvmh = get_quotes(symbol='MC', start_date=start, end_date=end,
                          data_source='yahoo')
        data_readers = [brown_forman, lvmh]
        names = ['Brown Forman', 'LVMH']

    elif sector == 'health':
        # Health
        united_health = get_quotes(symbol='UNH', start_date=start, end_date=end,
                                   data_source='yahoo')
        anthem = get_quotes(symbol='ANTM', start_date=start, end_date=end,
                            data_source='yahoo')
        medtronic = get_quotes(symbol='MDT', start_date=start, end_date=end,
                               data_source='yahoo')
        johnson_and_johnson = get_quotes(symbol='JNJ', start_date=start, end_date=end,
                                         data_source='yahoo')
        becton_dickinson = get_quotes(symbol='BDX', start_date=start, end_date=end,
                                      data_source='yahoo')
        abbott_laboratories = get_quotes(symbol='ABT', start_date=start, end_date=end,
                                         data_source='yahoo')
        data_readers = [united_health, anthem, medtronic, johnson_and_johnson,
                        becton_dickinson, abbott_laboratories]
        names = ['United Health', 'Anthem', 'Medtronic', 'Johnson & Johnson',
                 'Becton Dickinson', 'Abbott Labs']

    elif sector == 'pharma':
        # Health
        pfizer = get_quotes(symbol='UNH', start_date=start, end_date=end,
                            data_source='yahoo')
        abbvie = get_quotes(symbol='ABBV', start_date=start, end_date=end,
                            data_source='yahoo')
        lily = get_quotes(symbol='LLY', start_date=start, end_date=end,
                          data_source='yahoo')
        data_readers = [pfizer, abbvie]
        names = ['Pfizer', "AbbVie", "Elli Lily"]

    elif sector == 'energy':
        # Energy
        linde = get_quotes(symbol='LIN', start_date=start, end_date=end,
                           data_source='yahoo')
        exxon = get_quotes(symbol='XOM', start_date=start, end_date=end,
                           data_source='yahoo')
        chevron = get_quotes(symbol='CVX', start_date=start, end_date=end,
                             data_source='yahoo')
        data_readers = [linde, exxon, chevron]
        names = ['Linde', 'Exxon', 'Chevron']

    elif sector == 'industry':
        # Heavy Industry
        united_technologies = get_quotes(symbol='UTX', start_date=start, end_date=end,
                                         data_source='yahoo')
        mmm = get_quotes(symbol='MMM', start_date=start, end_date=end,
                         data_source='yahoo')
        data_readers = [united_technologies, mmm]
        names = ['United Technologies', "3M"]

    elif sector == 'comms':
        # Communications
        att = get_quotes(symbol='T', start_date=start, end_date=end,
                         data_source='yahoo')
        verizon = get_quotes(symbol='VZ', start_date=start, end_date=end,
                             data_source='yahoo')
        data_readers = [att, verizon, ]
        names = ['AT&T', 'Verizon', ]

    elif sector == 'entr':
        # Entertainment
        disney = get_quotes(symbol='DIS', start_date=start, end_date=end,
                            data_source='yahoo')
        data_readers = [disney, ]
        names = ['Disney', ]

    elif sector == 'index':
        # Index
        sp500_1 = get_quotes(symbol='^GSPC', start_date=start, end_date=end,
                             data_source='yahoo')
        sp500_2 = get_quotes(symbol='SPY', start_date=start, end_date=end,
                             data_source='yahoo')
        data_readers = [sp500_1, sp500_2]
        names = ['S&P 500 ^GSPC', 'S&P 500 SPY', ]

    plt.figure()
    for i, data_reader in enumerate(data_readers):
        reader, meta_info = data_reader

        if reader is None:
            continue

        info = ', '.join([f'{key} = {meta_info[key]}' for key in meta_info])
        names[i] = names[i] + ': ' + info

        opening_prices = reader['Open'].values

        dates_inds = reader.index.to_list()
        dates_inds = [str(date).split(' ')[0] for date in dates_inds]

        plt.figure(1)
        plt.plot(dates_inds, opening_prices)

    plt.legend(names)
    plt.xticks(rotation=90)
    plt.title(f"Shares Prices - {start} to {end}")
    plt.show()
