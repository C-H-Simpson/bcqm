import config_public
import BCQM_public
import helper
import matplotlib.pyplot as plt
import dataprocessing
import numpy as np
import pandas as pd

#%%
x = helper.open_pickle('./data/example_model.txt')
y = helper.open_pickle('./data/example_observed.txt')

#%%
"""
Version 1
"""
# Get historical data x (predictor model) and y (target observation) by overlapping dates
x_hist, y_hist = dataprocessing.intersect(x, y)

# Check dates (1979 - 2017 will be the calibration period)
x_hist.index
y_hist.index

# Get future model data
x_fut = x[str(x_hist.index[-1].year + 1):'2050']
x_fut.index

# We use quantile mapping bias correction to get future projection y_fut for these dates.

# #%%
# """
# Version 2
# """
# x = x[str(y.index[0].year):'2050']
#
# conf = config_public.Config(MODE='detrended_pdf',
#                             n_quantiles=80,
#                             x=x,
#                             y=y,
#                             )

#%%
conf = config_public.Config(MODE='detrended_polynomial',
                            n_quantiles=80,
                            x_hist=x_hist,
                            y_hist=y_hist,
                            x_fut=x_fut,
                            )

#%%

# Get projection
bc = BCQM_public.BiasCorrection(conf)
y_fut = bc.get_projection(RUN_KFOLD=False)

# Run K-fold experiments (historical period, uses x_hist and y_hist). Set to False if you just want projection data.
y_fut.head()

# bc.x_test_pdf.head()
# bc.corrected_timeseries.head()

#%%
# Plot results

smooth_curves = False
smooth_window = 5
alpha = 0.7

plt.figure()
y_plot = helper.smooth_savgol(conf.x_hist.values.squeeze(), window=smooth_window) if smooth_curves else conf.x_hist.values.squeeze()
plt.plot(conf.x_hist.index,
         y_plot,
         label='Predictor model'
         )

y_plot = helper.smooth_savgol(conf.y_hist.values.squeeze(), window=smooth_window) if smooth_curves else conf.y_hist.values.squeeze()
plt.plot(conf.y_hist.index,
         y_plot,
         alpha=alpha,
         label='Observation'
         )

y_plot = helper.smooth_savgol(y_fut.values.squeeze(), window=smooth_window) if smooth_curves else y_fut.values.squeeze()
plt.plot(y_fut.index,
         y_plot,
         alpha=alpha,
         label='Projection'
         )

plt.xlabel("Time")
plt.ylabel("Maximum daily temperature")
plt.legend()
