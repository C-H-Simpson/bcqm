import numpy as np
# import datetime
# from datasim import create_data
# from collections import defaultdict
from . import helper
# import plot
import pandas as pd
from . import dataprocessing
import xarray as xr
# import settings


class Config:
    '''
    Configuration file

    FOR GP_REGRESSION - want to use n_quantiles=None
    '''

    def __init__(self,
                 MODE='detrended_polynomial',  # 'detrended_pdf', 'detrended_polynomial'
                 ALGORITHM='vanilla',  # regression_polynomial, regression_keras, regression_nn, regression_gp, None

                 x_hist=None,  # Manually provide historical x (predictor e.g. model)
                 y_hist=None,  # Manually provide historical y (target e.g. observational)
                 x_fut=None,   # Manually provide future x (predictor e.g. model)
                 x=None,
                 y=None,

                 USE_RATIO=False,

                 REINSERT_TREND=False,   # DEPRECATED corrected_timeseries to include original trend in y (not trend in x)
                 PROJECT_TREND=True,     # projected_timeseries becomes available where trend in x is used
                 detrend_order=2,
                 detrend_window_days=None,  # Default same as window_days
                 detrend_window_years=1,    # contruct pdf using ± years for detrending purposes
                 detrend_using=np.median,   # For MODE='detrended_pdf': detrend pdf using this as a trend fit
                 TREND_PDF_METHOD=1,        # For MODE='detrended_pdf': Method 1 or 2 for detrend_pdf (performance should be about the same)

                 SUBSAMPLE=False,
                 n_subsample_days=7,
                 BLOCKING_DATA=True,  # New data
                 NORMALISE=True,
                 results_type='all',        # Set to 'summer' for JJA RMSE summary
                 results_summerhighq=True,  # Get results for summer (JJA) upper tail

                 DATA='public',  # public, x_and_y, old (local), new (local), grid (local)
                 loc=None,
                 x_data=None,  # String
                 y_data=None,  # String

                 K=5,                 # Number of folds in K-fold validation
                 window_years='all',  # 'all' uses all training years to create PDF. e.g. 7 = ±3 years
                 window_days=31,      # e.g. 31 = ±15 days

                 regression_order=3,     # Polynomial regression complexity
                 #x_cols_i=0,             # DEPRECATED Select features list
                 x_cols=['Tx', 'quantile', 'day_of_year'],
                 y_name='Ty',  # Set to 'bias' if we want to learn bias instead of target variable

                 hidden_layers=(10, ),   # NN architecture
                 penalty=0.0001,         # NN L2 penalty
                 solver='adam',          # NN optimiser
                 learning_rate=0.001,    # NN learning rate
                 max_iter=1000,          # NN maximum epochs

                 keras_optimizer='adam',       # Keras - optimiser type
                 keras_n_nodes=10,             # Keras - hidden nodes in layer 1
                 keras_n_nodes2=None,          # Keras - hidden nodes in layer 2
                 keras_l2_lambda=0.001,        # Keras - L2 penalty
                 keras_verbose=0,              # Keras - print optimisation steps while training

                 gp_n_pseudopoints=200,        # GP - number of pseudopoints for sparse approximation
                 gp_n_subsample=None,          # GP - randomly subsample data before training (set to None, reduce pseudopoints instead)
                 gp_verbose=False,             # GP - print optimisation steps while training
                 gp_IVWA=True,                 # GP - use IVWA to obtain corrected timeseries
                 gp_include_original_q=False,

                 n_quantiles=80,
                 test_quantiles=100,
                 high_quantile_threshold=90,

                 verbose_dataprocessing=False,
                 ):

        # MODE: preprocessing detrend
        # ['timeseries', 'detrended_polynomial', 'detrended_md', 'detrended_md_roll', 'subtract_median_md']
        # Note: detrend_test is for testing polynomial detrending but adding bias in longterm trend back in as a correction
        self.MODE = MODE

        # ['vanilla', 'regression_polynomial']
        # Vanilla BC correction, polynomial regression
        self.ALGORITHM = ALGORITHM
        self.DATA = DATA
        self.GRID = True if self.DATA.startswith('grid') else False

        self.results_type = results_type
        self.results_summerhighq = results_summerhighq

        # -----------------------------------------------#
        #  Detrending settings                           #
        # -----------------------------------------------#

        if self.ALGORITHM == 'None':
            # print("Setting config.REINSERT_TREND = False because ALGORITHM is None")
            self.REINSERT_TREND = False
        else:
            self.REINSERT_TREND = REINSERT_TREND  # FOR RMSE IT DOESN'T MATTER

        self.PROJECT_TREND = PROJECT_TREND

        #if self.MODE is 'timeseries':
        #    self.PROJECT_TREND = False

        # If using multiple window days for md detrending (no window days works best - because it flattens all data...)
        self.detrend_window_days = window_days if detrend_window_days is None else detrend_window_days
        self.detrend_window_years = detrend_window_years
        # Specify polynomial degrees fit if detrending by subtracting long term trend. degrees=1 will fit linear trend
        self.polyfit_degrees = detrend_order
        self.detrend_using = detrend_using
        self.TREND_PDF_METHOD = TREND_PDF_METHOD

        # -----------------------------------------------#
        #  Regression settings                           #
        # -----------------------------------------------#

        #self.TF = TF  # Run Tensorflow
        self.NORMALISE = NORMALISE
        SUBSAMPLE = True if MODE == 'timeseries' else SUBSAMPLE  # Always subsample if we're not detrending data: keeping all original years as feature = cannot use set quantiles = too much data!
        self.SUBSAMPLE_REGRESSION = SUBSAMPLE  # Subsample PDF for training

        # Polynomial regression settings
        self.polyfit_order = regression_order  # Non-linear polynomial fitting order

        # Sklearn NN settings
        self.penalty = penalty
        self.hidden_layers = hidden_layers
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # Keras settings
        self.optimizer = keras_optimizer
        self.n_nodes = keras_n_nodes
        self.n_nodes2 = keras_n_nodes2
        self.l2_lambda = keras_l2_lambda
        self.keras_verbose = keras_verbose

        # GP settings
        self.gp_n_subsample = gp_n_subsample
        self.gp_n_pseudopoints = gp_n_pseudopoints
        self.gp_verbose = gp_verbose
        self.gp_IVWA = gp_IVWA
        self.gp_include_original_q = gp_include_original_q

        # Choose from which columns to use NOT USED IN ALGOS
        self.x_cols_list = [['Tx', 'quantile', 'day_of_year', 'year'],  # 0
                          ['Tx', 'quantile', 'day_of_year'],  # 1
                          ['Tx', 'quantile'],  # 2
                          ['Tx'],  # 3
                          ['quantile']]  # 4

        self.x_cols = x_cols
        self.y_name = y_name

        if ALGORITHM.startswith('regression'):
            print('Using feature set: {}'.format(self.x_cols))

        # -----------------------------------------------#
        #  Algorithm settings                            #
        # -----------------------------------------------#

        # Rolling window frames
        self.window_days = window_days  # e.g. 31 = ±15 days
        self.window_years = window_years  # e.g. 7 = ±3 years
        if self.window_days != self.detrend_window_days:
            print("Warning: used window_days not the same as detrend_window_days (window in pdf used to detrend)")

        # Subsample pdf every n days to reduce training data
        # Used only for regression for now!
        self.SUBSAMPLE = SUBSAMPLE
        self.n_subsample_days = n_subsample_days

        # Cross validation settings
        self.K = K

        # Training quantiles and test quantiles
        self.n_quantiles = n_quantiles

        # High quantile threshold for RMSE
        self.high_quantile_threshold = high_quantile_threshold

        if self.ALGORITHM == 'regression_gp':
            print("Setting to n_quantiles='None' because MODE is 'regression_gp'".format(n_quantiles))
            self.n_quantiles = None

        if 'year' in self.x_cols:
            print("Setting to n_quantiles='None' because x_cols contains 'year'".format(n_quantiles))
            self.n_quantiles = None

        self.test_quantiles = test_quantiles

        # Bias as ratio (otherwise absolute difference)
        self.USE_RATIO = USE_RATIO

        self.verbose_dataprocessing = verbose_dataprocessing

        #------------------------------------------------#
        #  Load data                                     #
        #------------------------------------------------#

        self.var = 'tasmax'  # Variable
        self.rcp = 85                    # 45 or 85
        self.type = 'timeseries' + str(self.rcp)  # timeseries type

        # ['HadGEM2-ES', 'GFDL-ESM2G', 'CanESM2']  # old
        # ['HadGEM2-CC', 'CMCC-CM', 'MPI-ESM-MR']  # new
        self.x_data = 'HadGEM2-ES' if x_data is None else x_data

        # ['era', 'HadGEM2-ES', 'GFDL-ESM2G', 'CanESM2'], # ['HadGEM2-CC', 'CMCC-CM', 'MPI-ESM-MR']  # new
        self.y_data = 'era' if y_data is None else y_data

        # ['london', 'cairo'] # old
        # ['london', 'madrid', 'cairo', 'beijing', 'tokyo']  # new
        self.loc = 'london' if loc is None else loc

        # Old data (local)
        #-------------------
        if DATA == 'old':
            self.x = helper.open_pickle('./data/df/{}_{}_{}_{}.txt'.format(self.var, self.x_data, 'timeseries{}'.format(self.rcp), self.loc))
            if self.y_data == 'era':
                self.y = helper.open_pickle('./data/df/{}_{}_{}.txt'.format(self.var, 'era', self.loc))
                self.x = self.x.loc[str(self.y.index[0].year):str(self.y.index[-1].year)]  # Adjust GCM years to match ERA
            else:
                self.y = helper.open_pickle('./data/df/{}_{}_{}_{}.txt'.format(self.var, self.y_data, 'timeseries{}'.format(self.rcp), self.loc))
                #self.x = self.x.loc[str(self.y.index[0].year):str(self.y.index[-1].year)]

        # NEW DATA - /data/df/new_... (local)
        #------------------------------------
        elif DATA == 'new':
            self.x = helper.open_pickle('./data/df/new_{}_{}_{}_{}.txt'.format(self.var, self.x_data, 'timeseries{}'.format(self.rcp), self.loc))
            if self.y_data == 'era':
                self.y = helper.open_pickle('./data/df/new_{}_{}_{}.txt'.format(self.var, 'era', self.loc))
                self.x = self.x.loc[str(self.y.index[0].year):str(self.y.index[-1].year)]  # Adjust GCM years to match ERA
            else:
                self.y = helper.open_pickle('./data/df/new_{}_{}_{}_{}.txt'.format(self.var, self.y_data, 'timeseries{}'.format(self.rcp), self.loc))

            # Make sure data column is labelled
            try:
                self.x.set_axis(['data'], axis='columns', inplace=True)
                self.y.set_axis(['data'], axis='columns', inplace=True)
            except:
                pass

        # Gridded data for 1st yr report (local)
        #----------------------------------------
        elif DATA == 'grid_europe':
            self.x = xr.open_dataarray('./data/window_{}_processed.nc'.format(self.x_data)).load()
            self.y = xr.open_dataarray('./data/window_{}_processed.nc'.format(self.y_data)).load()

        #  Public (x_hist, y_hist and x_fut dataframes provided)
        #--------------------------------------------------------
        elif DATA == 'public':
            if x_hist is None or x_fut is None or y_hist is None:
                raise ValueError("Please provide x_hist, y_hist and x_fut, or x and y")
            self.x_hist = x_hist
            self.y_hist = y_hist
            self.x_fut = x_fut

            assert len(self.x_hist) == len(self.y_hist)

            self.x = pd.concat((x_hist, x_fut))
            if sum(self.x.index.duplicated()) > 0:
                raise ValueError("Found duplicated index in x_hist and x_fut.")

            self.y = y_hist

        elif DATA == 'x_and_y':
            if x is None or y is None:
                raise ValueError("Please provide x and y")
            self.x = x
            self.y = y

        # Save original copies of files before detrending
        print("Original data loaded.")
        self.x_original = self.x.copy()
        self.y_original = self.y.copy()

        # Make sure X and Y are the same size
        if DATA != 'public' and DATA != 'x_and_y':
            assert len(self.x) == len(self.y)

        # List of all years used in x and y
        try:
            ind = self.x.index
        except:
            ind = dataprocessing.da_to_df(self.x[:, 0, 0]).index
        self.years_all = np.unique([date.year for date in ind])

        #------------------------------------------------#
        #  Detrending: subtracting long term trends      #
        #------------------------------------------------#
        """
        Output:
        self.x
        self.y
        self.trend_x
        self.trend_y
        """

        if self.MODE != 'timeseries':

            from dataprocessing import Df_to_df, Detrend
            df_to_df = Df_to_df()
            detrend = Detrend(MODE=self.MODE, polyfit_degrees=self.polyfit_degrees)  # Create detrend instance with MODE

            def get_detrended(input):

                print("Getting trend data, config.MODE = {}...".format(self.MODE))

                if self.MODE == 'detrended_md_roll':
                    # Roll window_days for all days
                    # print("Creating PDF for detrend info...")
                    pdf = df_to_df.create_pdf(input,
                                              window_days=self.detrend_window_days,
                                              window_years=1,
                                              PRESERVE_DATES=False,
                                              PRESERVE_EDGES=True,
                                              verbose=verbose_dataprocessing,
                                              )
                    output, trend = detrend.detrend_by_md(input, df_pdf=pdf)  # Output is detrended

                elif self.MODE == 'detrended_md':
                    output, trend = detrend.detrend_by_md(input)  # Output is detrended

                elif self.MODE == 'subtract_median_md':
                    output, trend = detrend.subtract_median_md(input)  # Output is detrended

                elif self.MODE == 'detrended_polynomial':
                    output, trend = detrend.detrend_df(input)  # Output is detrended

                elif self.MODE == 'detrended_pdf':
                    # NEW: use PDF mean or median to construct trend for each day of year, detrend on pdf before QM is performed (not raw data like in detrend_md)
                    # print("Creating PDF for detrend info...")
                    pdf = df_to_df.create_pdf(input,
                                              window_days=self.detrend_window_days,
                                              window_years=self.detrend_window_years,
                                              PRESERVE_DATES=False,
                                              PRESERVE_EDGES=True,
                                              verbose=verbose_dataprocessing,
                                              )
                    output, trend = detrend.detrend_by_pdf(input, df_pdf=pdf, detrend_using=detrend_using)  # Output is RAW DATA

                else:
                    raise ValueError("Invalid config.MODE")

                return output, trend


            #if self.MODE is 'detrended_md' and self.GRID:
            if self.GRID:

                print("Processing trend info on all grid...")
                x_ = self.x.copy()
                y_ = self.y.copy()
                x_trend = self.x.copy()
                y_trend = self.y.copy()

                for i in range(10):
                    print("Processing {}/10".format(i))
                    for j in range(10):
                        x_point = dataprocessing.da_to_df(self.x[:, i, j])
                        x_point, x_trend_point = get_detrended(x_point)
                        x_[:, i, j] = x_point.values.squeeze()
                        x_trend[:, i, j] = x_trend_point.values.squeeze()

                        y_point = dataprocessing.da_to_df(self.y[:, i, j])
                        y_point, y_trend_point = get_detrended(y_point)
                        y_[:, i, j] = y_point.values.squeeze()
                        y_trend[:, i, j] = y_trend_point.values.squeeze()

                self.x = x_
                self.y = y_
                self.trend_x = x_trend
                self.trend_y = y_trend

            else:

                print("Processing trend info on 1D data")
                print("Processing x")
                self.x, self.trend_x = get_detrended(self.x)
                print("Processing y")
                self.y, self.trend_y = get_detrended(self.y)

        print("Loaded config.")
