import matplotlib

matplotlib.use('TKAgg')

from src.io import anaylze_assets


sectors = {
    0: 'tech',
    1: 'finance',
    2: 'food',
    3: 'retail',
    4: 'lux',
    5: 'health',
    6: 'energy',
    7: 'industry',
    8: 'comms',
    9: 'entr',
    10: 'index',
    11: 'pharma',
}

# start = datetime.datetime(2012, 1, 1)
# end = datetime.datetime(2020, 6, 4)

# plot_sector(sector=sectors[0], start=start, end=end)
# plot_sector(sector=sectors[1], start=start, end=end)
# plot_sector(sector=sectors[3], start=start, end=end)
# plot_sector(sector=sectors[5], start=start, end=end)
# plot_sector(sector=sectors[6], start=start, end=end)
# plot_sector(sector=sectors[9], start=start, end=end)

n_years = 5
analysis_period_len = n_years * 365
data_source = 'yahoo'
metric2_analyze = 'Adj Close'
mini_period_len = 5
comp_desired_gain = 0.75
comp_desired_mean_gain = 0.80
start_date = None

anaylze_assets(analysis_period_len=analysis_period_len, data_source=data_source,
               metric2_analyze=metric2_analyze, mini_period_len=mini_period_len,
               desired_peak_gain=comp_desired_gain,
               desired_mean_gain=comp_desired_mean_gain)






