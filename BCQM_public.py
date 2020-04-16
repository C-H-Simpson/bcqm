"""
BCQM v4 - vanilla QM
"""

import numpy as np
import pandas as pd
import itertools
from scipy.stats import rankdata
from scipy.stats import ks_2samp

from . import dataprocessing
from .dataprocessing import Df_to_df
df_to_df = Df_to_df()

# try:
#     """ DOES NOT WORK ON JASMIN - tensorflow cannot be installed """
#
#     from keras.models import Sequential, load_model
#     from keras.layers import Dense, Dropout
#     #from keras.wrappers.scikit_learn import KerasRegressor
#     from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#     from keras.regularizers import l2
#     import gpflow
#     from gpflow.kernels import RBF, Periodic
#
# except:
#     print("WARNING: could not import keras and gpflow")


class BiasCorrection:
    '''
    Bias correction
    '''

    def __init__(self,
                 config,
                 ):
        """
        Import config as argument
        """

        self.config = config

        if config.MODE == 'detrended_pdf' or config.MODE == 'detrended_polynomial':
            self.trend_x = config.trend_x
            self.trend_y = config.trend_y
        else:
            self.trend_x = None
            self.trend_y = None

    """
    -------------------------------------------------------------------------
    Main experiments (get_projection, k-fold, PS)
    -------------------------------------------------------------------------
    """

    def get_projection(self,
                       RUN_KFOLD=False,
                       ):
        """
        """

        self.train(self.config.x_hist, self.config.y_hist)
        self.correct_timeseries(self.config.x_fut, PROJECT_TREND=True)

        if RUN_KFOLD:
            print("\nRunning K-fold cross validation experiment. Set RUN_KFOLD=False to skip.")
            self.k_fold_(self.config.x_hist, self.config.y_hist)
            print("\nK-fold scores: \nRMSE = {}, \nRMSE (summer) = {}, \nKS test score = {}, \nKS test score (summer) = {}".format(self.rmse_final, self.rmse_summer, self.ks_mean_score, self.ks_summer_score))

            self.results = [self.rmse_final, self.rmse_summer, self.ks_mean_score, self.ks_summer_score]
            print("(Access at self.results)")

        return self.projected_timeseries


    def run_experiment_1d(self,
                          experiment='perfect sibling',
                          x=None,
                          y=None,
                          i_grid=None,
                          j_grid=None,
                          ):
        """
        experiment='perfect sibling' or 'k-fold'
        Warning: get_residuals returns all x and its errors (large files!)
        """

        x = self.config.x if x == None else x
        y = self.config.y if y == None else y
        self.i_grid = i_grid
        self.j_grid = j_grid

        # self.get_residuals = get_residuals
        # self.ks_get_p = ks_get_p  # Get p-value instead of KS score

        if experiment == 'perfect sibling':
            self.perfect_sibling_(x, y)

        elif experiment == 'k-fold':
            self.k_fold_(x, y)

        else:
            raise ValueError("Please enter valid experiment name ('perfect sibling' or 'k-fold')")


    def perfect_sibling_(self, x, y):
        """
        Scores:
        - self.ks_mean_p
        - self.ks_summer_p
        - self.ks_mean_score
        - self.ks_summer_score

        - self.rmse_all: RMSE (time of year, quantile)
        - self.rmse_final: one value RMSE (average all years and quantiles)
        - self.rmse_summer: one value RMSE (average JJA and all quantiles)
        """

        # (1) Split timeseries into train and test data
        self.x_train_ = x.loc['1979':'2017']
        self.y_train_ = y.loc['1979':'2017']
        self.x_test_ = x.loc['2018':'2100']
        self.y_test_ = y.loc['2018':'2100']

        if self.config.ALGORITHM != 'None':
            # (3) Calculate bias from training data
            self.train(self.x_train_, self.y_train_)

            # (4) Obtain corrected timeseries for test data
            self.correct_timeseries(self.x_test_)

            # (5) Get RMSE using same rolling window as training
            self.get_rmse(
                corrected_timeseries=self.corrected_timeseries, y_test_=self.y_test_)

        # Do not apply correction if config.ALGORITHM is None
        else:
            print("config.ALGORITHM is None: Not applying correction")
            self.get_rmse(
                corrected_timeseries=self.x_test_, y_test_=self.y_test_)

            # Not correcting timeseries so save "projected_timeseries" as itself
            if self.config.GRID:  # If we are using gridded data
                self.projected_timeseries = dataprocessing.da_to_df(self.config.y_original[:, self.i_grid, self.j_grid]).loc['2018':'2100']
            else:
                self.projected_timeseries = dataprocessing.da_to_df(self.config.y_original.loc['2018':'2100'])


    def k_fold_(self, x, y):

        """
        Scores:
        - self.ks_mean_p
        - self.ks_summer_p
        - self.ks_mean_score
        - self.ks_summer_score

        - self.rmse_all: RMSE (time of year, quantile)
        - self.rmse_final: one value RMSE (average all years and quantiles)
        - self.rmse_summer: one value RMSE (average JJA and quantiles)
        """

        self.config.PROJECT_TREND = False  # Do not project trend if performing K-fold testing
        self.config.verbose_dataprocessing = False


        # Save results for each K
        rmse_all_k = []
        rmse_final_k = []
        rmse_summer_k = []

        ks_mean_p_k = []
        ks_mean_score_k = []
        ks_summer_p_k = []
        ks_summer_score_k = []

        # Residual data
        #if self.get_residuals:
        residuals_list_k = []
        y_true_test_list_k = []
        y_pred_test_list_k = []

        # Calculate length in year - not using index 0 as it might not have full length year
        dummy_yr = np.unique([date.year for date in x.index])[1]
        len_in_year = len(x.loc[str(dummy_yr)].index)

        # How many years in each test window
        yr_split = int(np.round(len(y) / len_in_year / self.config.K))
        years_all = np.unique([date.year for date in y.index])

        starts = np.arange(0, len(years_all) + 1, yr_split)

        for k in range(self.config.K):

            print("K = {}/{}".format(k + 1, self.config.K))

            start_idx = starts[k]
            end_idx = start_idx + yr_split
            test_years = years_all[start_idx:end_idx]

            # (1) Split timeseries into train and test data
            self.x_train_ = x[~x.index.year.isin(test_years)]
            self.y_train_ = y[~y.index.year.isin(test_years)]
            self.x_test_ = x[x.index.year.isin(test_years)]
            self.y_test_ = y[y.index.year.isin(test_years)]

            if self.config.ALGORITHM != 'None':
                # (3) Calculate bias from training data
                # print(" - Calculating bias from training data...")
                self.train(self.x_train_, self.y_train_)

                # (4) Obtain corrected timeseries for test data
                # print(" - Correcting timeseries...")
                self.correct_timeseries(self.x_test_, PROJECT_TREND=False)

                # (5) Get RMSE using same rolling window as training
                # print(" - Calculating RMSE with specified windows...")
                self.get_rmse(
                    corrected_timeseries=self.corrected_timeseries, y_test_=self.y_test_, verbose_results=False)

            # Do not apply correction if config.ALGORITHM is None
            else:
                print("config.ALGORITHM is None: Not applying correction, \n WARNING: self.projected_timeseries is None in K-fold experiment")

                self.get_rmse(
                    corrected_timeseries=self.x_test_, y_test_=self.y_test_, verbose_results=False)

                self.projected_timeseries = None  # No projection

            # (6) Save result for each k in dicts
            print("     RMSE for this k = {}".format(self.rmse_final))

            # SAVE SCORES
            rmse_all_k.append(self.rmse_all)
            rmse_final_k.append(self.rmse_final)
            rmse_summer_k.append(self.rmse_summer)

            ks_mean_p_k.append(self.ks_mean_p)
            ks_summer_p_k.append(self.ks_summer_p)
            ks_mean_score_k.append(self.ks_mean_score)
            ks_summer_score_k.append(self.ks_summer_score)

            #if self.get_residuals:
            # Long list of values
            residuals_list_k.append(self.residuals_list_all)
            y_true_test_list_k.append(self.y_true_test_list_all)
            y_pred_test_list_k.append(self.y_pred_test_list_all)

        # K LOOP END

        # Average results for all K
        self.rmse_all = np.mean(rmse_all_k, axis=0)
        self.rmse_final = np.mean(rmse_final_k)
        self.rmse_summer = np.mean(rmse_summer_k)

        self.ks_mean_p = np.mean(ks_mean_p_k)
        self.ks_summer_p = np.mean(ks_summer_p_k)
        self.ks_mean_score = np.mean(ks_mean_score_k)
        self.ks_summer_score = np.mean(ks_summer_score_k)

        # Residuals
        #if self.get_residuals:
        self.residuals_list_all = np.array(residuals_list_k)
        self.y_true_test_list_all = np.array(y_true_test_list_k)
        self.y_pred_test_list_all = np.array(y_pred_test_list_k)


    """
    -------------------------------------------------------------------------
    Functions used in k-fold experiments
    -------------------------------------------------------------------------
    """

    def train(self, x_train_=None, y_train_=None):
        """
        Calculate bias from training data
        Inputs:
            - Training data timeseries df, x_train_ and y_train_
        Updated class variables:
            - self.x_train_pdf
            - self.y_train_pdf
        Output:
            - Mean bias: mean_bias
        """

        if self.config.ALGORITHM.startswith('regression'):
            print("X = {}".format(self.config.x_cols))

        x_train_ = self.x_train_ if x_train_ is None else x_train_
        y_train_ = self.y_train_ if y_train_ is None else y_train_

        # Overwrite class training data
        self.x_train_ = x_train_
        self.y_train_ = y_train_

        # ------------- #
        #  Create PDF   #
        # ------------- #

        print("Generating pdf for training data...")
        # if 'myVar' in locals(): <-- check if variables exists in local (self. always local)
        self.x_train_pdf = df_to_df.create_pdf(x_train_,
                                               window_days=self.config.window_days,
                                               window_years=self.config.window_years,
                                               PRESERVE_DATES=True,  # Changed to True in case we need it for MODE='timeseries' regression
                                               detrend_trend=self.trend_x,
                                               verbose=self.config.verbose_dataprocessing,
                                               )

        self.y_train_pdf = df_to_df.create_pdf(y_train_,
                                               window_days=self.config.window_days,
                                               window_years=self.config.window_years,
                                               PRESERVE_DATES=True,
                                               detrend_trend=self.trend_y,
                                               verbose=self.config.verbose_dataprocessing,
                                               )

        # -------------- #
        #  Train model   #
        # -------------- #

        print("Training model...")
        # Vanilla BC
        if not self.config.ALGORITHM.startswith('regression'):
            _ = self.train_qm(self.x_train_pdf, self.y_train_pdf)

        # # Regression
        # if self.config.ALGORITHM.startswith('regression'):
        #     print("Using regression model")
        #     _ = self.train_regression(self.x_train_pdf, self.y_train_pdf)


    def correct_timeseries(self,
                           x_test_=None,
                           model=None,
                           PROJECT_TREND=None,  # Manual option
                           ):

        """
        Get corrected timeseries

        Input:
            - Test data timeseries df: x_test_
            Default: use test data

        Updated class variables:
            - self.x_train_pdf
            - self.y_train_pdf

        Output
            - self.corrected_timeseries
        """

        print("Correcting x_test timeseries...")

        x_test_ = self.x_test_ if x_test_ is None else x_test_

        # Overwrite class data
        self.x_test_ = x_test_

        model = self.model if model is None else model
        if PROJECT_TREND is None:
            PROJECT_TREND = self.config.PROJECT_TREND

        # ------------------------------------------ #
        #  Get adjusted trend from model for future  #
        # ------------------------------------------ #

        if PROJECT_TREND:
            if self.config.MODE == 'detrended_pdf':
                _ = self.get_projected_trend()   # Get adjusted trend for future from model, self.trend_pdf_adjusted
                if self.config.TREND_PDF_METHOD == 1:
                    trend_pdf_adjusted = self.trend_pdf_adjusted
                    # trend_pdf_adjusted = self.trend_pdf_adjusted if trend_pdf_adjusted is None else trend_pdf_adjusted
                else:
                    trend_pdf_adjusted = None

            elif self.config.MODE == 'detrended_polynomial':
                _ = self.get_projected_trend()
                trend_pdf_adjusted = None

            else:
                trend_pdf_adjusted = None
        else:
            trend_pdf_adjusted = None

        # ------------ #
        #  Create PDF  #
        # ------------ #

        print("Generating pdf for test data...")
        self.x_test_pdf = df_to_df.create_pdf(x_test_,
                                              window_days=self.config.window_days,
                                              window_years=self.config.window_years,
                                              PRESERVE_DATES=True,
                                              detrend_trend=self.trend_x,
                                              trend_pdf=trend_pdf_adjusted,  # Use if TREND_PDF_METHOD == 1
                                              verbose=self.config.verbose_dataprocessing,
                                              )  # Not subsampling for test

        # ----------------------------- #
        #  Correct using trained model  #
        # ----------------------------- #
        # Use lookup vanilla BC
        if not self.config.ALGORITHM.startswith('regression'):
            corrected_all = self.correct_qm(self.x_test_pdf)
        # Use regression model
        if self.config.ALGORITHM.startswith('regression'):
            print("Applying regression model")
            corrected_all = self.correct_regression(self.x_test_pdf, model)

        # ------------------------------------------------------------------------------- #
        #  Reinsert original trend to raw corrections before taking average for each day  #
        # ------------------------------------------------------------------------------- #
        if self.config.MODE == 'detrended_pdf' and self.config.TREND_PDF_METHOD == 1:
            if 'trend_pdf' in corrected_all.columns:
                corrected_all['data'] = corrected_all['data'] + corrected_all['trend_pdf']

        # -------------------------- #
        #  Get average for each day  #
        # -------------------------- #

        # Group all corrected values for each date and get average
        group = corrected_all.groupby('date_raw')
        corrected_group = group['data'].apply(np.array).reset_index()

        # --------------------------------- #
        #  Calculate average for each date  #
        # --------------------------------- #


        vals = corrected_group['data'].apply(np.mean)
        self.corrected_timeseries_ = pd.DataFrame(
            index=corrected_group['date_raw'], data={'data': vals.values}
        )

        self.corrected_timeseries = self.corrected_timeseries_

        print("Done: adjusted result at self.corrected_timeseries (may be in detrended form; not necessarily projection)")

        # -------------------------- #
        #  Get projected trend       #
        # -------------------------- #

        if PROJECT_TREND:
            _ = self.get_projected_timeseries()  # Uses self.corrected_timeseries


    def get_projected_trend(self):
        """
        Works with self.correct_timeseries()
        Get projected trend
        """

        if self.config.MODE == 'detrended_polynomial':

            # Get the end of the observation trends (end of historical)
            trend_y_train, _ = dataprocessing.intersect(self.config.trend_y, self.x_train_)  # tr_y
            trend_y_train_grouped = trend_y_train.groupby([trend_y_train.index.month, trend_y_train.index.day])['data'].apply(np.array)
            trend_y_train_grouped = trend_y_train_grouped.reset_index(drop=True)
            trend_tail = np.array([x[-1] for x in trend_y_train_grouped.squeeze()])

            # Get the start of model trends (beginning of future)
            trend_x_test, _ = dataprocessing.intersect(self.config.trend_x, self.x_test_)  # tr_x
            trend_x_test_grouped = trend_x_test.groupby([trend_x_test.index.month, trend_x_test.index.day])['data'].apply(np.array)
            trend_x_test_grouped = trend_x_test_grouped.reset_index(drop=True)
            trend_head = np.array([x[0] for x in trend_x_test_grouped.squeeze()])

            # Match end of observed historical to start of model future
            trend_x_test_grouped_fixed = trend_x_test_grouped + (trend_tail - trend_head)

            df = trend_x_test.copy()
            df['date_raw'] = df.index
            df_date_raw = df.groupby([df.index.month, df.index.day])['date_raw'].apply(np.array)
            vals = np.reshape([np.stack(i) for i in trend_x_test_grouped_fixed], (-1,))
            date_raw = np.reshape([np.stack(i) for i in df_date_raw], (-1,))

            trend_x_test_adjusted = pd.DataFrame(index=date_raw, data={'data':vals}).sort_index()

            self.trend_pdf_adjusted = trend_x_test_adjusted


        elif self.config.MODE == 'detrended_pdf':

            # Get the end of the observation trends (end of historical)
            trend_y_train, _ = dataprocessing.intersect(self.config.trend_y, self.x_train_)
            trend_y_train_grouped = trend_y_train.groupby([trend_y_train.index.month, trend_y_train.index.day])['data'].apply(np.array)
            trend_y_train_grouped = trend_y_train_grouped.reset_index(drop=True)
            trend_tail = np.array([x[-1] for x in trend_y_train_grouped.squeeze()])

            # Get the start of model trends (beginning of future)
            trend_x_test, _ = dataprocessing.intersect(self.config.trend_x, self.x_test_)
            trend_x_test_grouped = trend_x_test.groupby([trend_x_test.index.month, trend_x_test.index.day])['data'].apply(np.array)
            trend_x_test_grouped = trend_x_test_grouped.reset_index(drop=True)
            trend_head = np.array([x[0] for x in trend_x_test_grouped.squeeze()])

            # Match end of observed historical to start of model future
            trend_x_test_grouped_fixed = trend_x_test_grouped + (trend_tail - trend_head)

            # Get back original df format
            df = trend_x_test.copy()
            df['date_raw'] = df.index
            df_date_raw = df.groupby([df.index.month, df.index.day])['date_raw'].apply(np.array)
            vals = np.reshape([np.stack(i) for i in trend_x_test_grouped_fixed], (-1,))
            date_raw = np.reshape([np.stack(i) for i in df_date_raw], (-1,))
            trend_x_test_adjusted = pd.DataFrame(index=date_raw, data={'data':vals}).sort_index()

            self.trend_pdf_adjusted = trend_x_test_adjusted

        else:
            print("WARNING: get_projected_trend() (not projected timeseries) was passed but config.MODE is not compatible.")
            self.trend_pdf_adjusted = None

        return self.trend_pdf_adjusted


    def get_projected_timeseries(self,
                                 ):

        print("Obtaining projected timeseries...")

        if self.config.MODE == 'timeseries':
            self.projected_timeseries = self.corrected_timeseries

        elif self.config.MODE == 'detrended_pdf':
            if self.config.TREND_PDF_METHOD == 1:
                self.projected_timeseries = self.corrected_timeseries

            elif self.config.TREND_PDF_METHOD == 2:

                y_test_corrected_pdf_day = df_to_df.create_pdf(self.corrected_timeseries,
                                                               window_days=self.config.detrend_window_days,
                                                               window_years=self.config.detrend_window_years,
                                                               PRESERVE_DATES=True,
                                                               verbose=self.config.verbose_dataprocessing,
                                                               )
                y_test_corrected_pdf_day['data'] = y_test_corrected_pdf_day['data'] + self.trend_pdf_adjusted.values.squeeze()
                df_new = df_to_df.unnesting(y_test_corrected_pdf_day, ['data', 'date_raw']).set_index('date_raw')
                group = df_new.groupby('date_raw')

                projected_group = group['data'].apply(np.array).reset_index()
                vals = projected_group['data'].apply(np.mean)
                self.projected_timeseries = pd.DataFrame(
                    index=projected_group['date_raw'], data={'data': vals.values}
                )

        elif self.config.MODE == 'detrended_polynomial':

            self.projected_timeseries = self.corrected_timeseries.copy()
            self.projected_timeseries['data'] = self.corrected_timeseries['data'] + self.trend_pdf_adjusted.values.squeeze()

        print("Done: projected timeseries at self.projected_timeseries")

        return self.projected_timeseries


    def get_rmse(self,
                 corrected_timeseries=None,
                 y_test_=None,
                 verbose_results=True):

        """
        Calculate test scores
        Uses self.corrected_timeseries and compares against self.y_test_pdf

        Inputs:
            - Corrected timeseries OR Corrected timeseries PDF: corrected_timeseries or y_test_corrected_pdf
            - True test timeseries: y_test_

        Updated class variables:
            - self.y_test_pdf
            - self.y_test_corrected_pdf (if not provided)

        Outputs
            Scores:
            - self.ks_mean_p
            - self.ks_summer_p
            (- self.ks_results: [ks_mean_p, ks_summer_p])
            - self.ks_mean_score
            - self.ks_summer_score
            (- self.ks_results_: [ks_mean_score, ks_summer_score])

            - self.rmse_all: RMSE (time of year, quantile)
            - self.rmse_final: one value RMSE (average all years and quantiles)
            - self.rmse_summer: one value RMSE (average JJA and quantiles)

            Residuals:
            - self.residuals_list_all: list of residuals in line with
            - self.y_true_test_list_all

        """
        print("Comparing corrected timeseries to y_test (truth):")

        y_test_ = self.y_test_ if y_test_ is None else y_test_
        corrected_timeseries = self.corrected_timeseries if corrected_timeseries is None else corrected_timeseries

        # Overwrite class data
        self.y_test_ = y_test_
        self.corrected_timeseries = corrected_timeseries

        # Ignore the variance here
        corrected_timeseries = corrected_timeseries[['data']]

        # Check length of corrected timeseries = length of original timeseries fair comparison
        # If not, match length
        if len(corrected_timeseries) != len(y_test_):
            print("WARNING: \
                  Corrected timeseries does not match length of original timeseries. \n \
                  This may be due to subsampling. \n \
                  Matching dates for fair comparison...")
            corrected_timeseries, y_test_ = dataprocessing.intersect(
                corrected_timeseries, y_test_)

        # -------------------------------- #
        #  Compare corrected vs. truth     #
        # -------------------------------- #
        # (2) Get PDF of corrected timeseries and y_test timeseries

        #if y_test_corrected_pdf is None:

        if self.config.PROJECT_TREND:
            if self.config.MODE == 'detrended_pdf' and self.config.TREND_PDF_METHOD == 1:
                detrend_trend = self.trend_pdf_adjusted
            else:
                detrend_trend = None
        else:
            detrend_trend = None

        print("Generating pdf for adjusted & truth data...")
        self.y_test_corrected_pdf = df_to_df.create_pdf(corrected_timeseries,
                                                        window_days=self.config.window_days,
                                                        window_years=self.config.window_years,
                                                        detrend_trend=detrend_trend,
                                                        PRESERVE_DATES=False,
                                                        verbose=self.config.verbose_dataprocessing,
                                                        )

        self.y_test_pdf = df_to_df.create_pdf(y_test_,
                                              window_days=self.config.window_days,
                                              window_years=self.config.window_years,
                                              detrend_trend=self.trend_y,
                                              PRESERVE_DATES=False,
                                              verbose=self.config.verbose_dataprocessing,
                                              )

        print("Calculating scores...")

        # (3) Compare 2 PDFs and obtain residuals and RMSE
        df_residuals = pd.DataFrame(
            index=self.y_test_pdf.index, columns=['residuals'])
        residuals_list = []  # (y* - y) for each data
        y_true_test_list = []
        y_pred_test_list = []

        # KS test results
        ks_results = []   # p value
        ks_results_ = []  # KS score

        for t in range(len(self.y_test_pdf)):

            # Corrected
            arr_y_corrected = self.y_test_corrected_pdf.iloc[t]['data']
            _, pred_y = self.percentiles(
                arr_y_corrected, n_quantiles=self.config.test_quantiles)

            # Truth
            arr_y_test = self.y_test_pdf.iloc[t]['data']
            _, vals_y = self.percentiles(
                arr_y_test, n_quantiles=self.config.test_quantiles)

            # Calculate RMSE and residuals
            residuals = pred_y - vals_y  # Discrepancies for each quantile separately

            # Save all data in dictionaries
            residuals_list.append(residuals)  # For unwinding later
            y_true_test_list.append(vals_y)
            y_pred_test_list.append(pred_y)

            df_residuals.at[self.y_test_pdf.index[t], 'residuals'] = residuals

            # KS test - get p-value
            ks = ks_2samp(pred_y, vals_y)
            ks_results.append(ks[1])  # p value
            ks_results_.append(ks[0])  # ks score


        # (1) Find RMSE average for each test years (time of year, quantile)
        if self.config.window_years == 'all':
            """ technically residuals (MAKE +VE) """
            rmse_all = df_residuals['residuals'].apply(np.array).values
            self.rmse_all = np.abs([np.stack(i) for i in rmse_all])

        else:
            rmse_all = df_residuals.groupby([df_residuals.index.month, df_residuals.index.day])[
                'residuals'].apply(np.array)
            rmse_all = rmse_all.apply(lambda lst: np.mean(lst ** 2) ** 0.5)
            self.rmse_all = np.array([np.stack(d) for d in rmse_all])

        # ------------- #
        #  Get scores   #
        #  ------------ #
        #if self.config.results_type is 'all':
        self.rmse_final = (np.mean(np.array(residuals_list) ** 2)) ** 0.5

        #elif self.config.results_type is 'summer':
        rmse_summer = np.reshape(self.rmse_all[151:243], (-1,))  # JJA
        self.rmse_summer = (np.mean(rmse_summer ** 2)) ** 0.5

        rmse_summerq = np.reshape(np.array(self.rmse_all[151:243])[:, self.config.high_quantile_threshold:], (-1,)) ###
        self.rmse_summerq = (np.mean(rmse_summerq ** 2)) ** 0.5

        # KS test results: p value
        self.ks_mean_p = np.mean(ks_results)
        self.ks_summer_p = np.mean(ks_results[151:243])
        self.ks_results = [self.ks_mean_p, self.ks_summer_p]

        # KS test results: ks score
        self.ks_mean_score = np.mean(ks_results_)
        self.ks_summer_score = np.mean(ks_results_[151:243])
        self.ks_results_ = [self.ks_mean_score, self.ks_summer_score]

        # ---------------------- #
        #  Values and residuals  #
        # ---------------------- #
        self.residuals_list_all = np.array(
            list(itertools.chain.from_iterable(residuals_list)))
        self.y_true_test_list_all = np.array(
            list(itertools.chain.from_iterable(y_true_test_list)))
        self.y_pred_test_list_all = np.array(
            list(itertools.chain.from_iterable(y_pred_test_list)))

        # ---------------------- #
        #  Get results in list   #
        #  --------------------- #

        self.results = [
                        self.rmse_all,
                        self.rmse_final,
                        self.rmse_summer,
                        self.ks_mean_p,
                        self.ks_summer_p,
                        self.ks_mean_score,
                        self.ks_summer_score,
        ]

        self.residuals = [
            self.y_true_test_list_all,
            self.y_pred_test_list_all,
            self.residuals_list_all
        ]

        if verbose_results:
            print("Access results:")
            print("self.results = [0: rmse_all, 1: rmse_final, 2: rmse_summer, 3: ks_p, 4: ks_p_summer, 5: ks_score, 6: ks_score_summer]")
            print("self.residuals = [0: y_true, 1: y_pred, 2: residuals]")
            #print("Done: Access results by .ks_results (p value all/summer), .ks_results_ (ks score all/summer), .rmse_all, .rmse_final, .residuals_list_all")

    """
    ================================================================
        Training methods
    ================================================================
    """

    def train_qm(self, x_train_pdf, y_train_pdf):

        df_d = pd.DataFrame(index=x_train_pdf.index, columns=['delta'])

        for t in range(len(x_train_pdf)):

            arr_x = x_train_pdf.iloc[t]['data']
            # Length of array = n_quantiles
            _, vals_q_x = self.percentiles(arr_x, n_quantiles=self.config.n_quantiles)

            arr_y = y_train_pdf.iloc[t]['data']
            _, vals_q_y = self.percentiles(arr_y, n_quantiles=self.config.n_quantiles)

            delta = vals_q_y / vals_q_x if self.config.USE_RATIO else vals_q_y - vals_q_x
            # Tobs / Tgcm (* Tx)

            df_d.at[x_train_pdf.index[t], 'delta'] = delta

        if self.config.window_years == 'all':
            df_d_grouped = df_d.values
        else:
            df_d_grouped = df_d.groupby([df_d.index.month, df_d.index.day])[
                'delta'].apply(np.mean).values

        # (day of year, quantile) averaged all years (number of keys)
        self.mean_bias = np.array([np.stack(d)
                                   for d in df_d_grouped]).squeeze()

        # Output model
        self.model = self.mean_bias

        return self.model


    def train_regression(self, x_train_pdf, y_train_pdf, algorithm=None):

        pass


    """
    ================================================================
        Correction methods (after training)
    ================================================================
    """

    def correct_qm(self, x_test_pdf=None):
        """
        Input: test pdf
        Output: corrected data in form:
        pd.DataFrame(data={'date_raw':sorted_dates, 'data':pred_y, 'var':var_y})
        """

        x_test_pdf = self.x_test_pdf if x_test_pdf is None else x_test_pdf

        # (2) Prepare to correct timeseries
        step = 100 / (self.config.n_quantiles)
        perc_train = np.arange(start=step, stop=100, step=step)

        # For labelling mean_bias by month-day so it's searchable when testing
        # dummy_yr = np.unique([date.year for date in self.x_train_pdf.index])[1]  # Not using index 0 as it might not have full length year
        # m_d = self.x_train_pdf.loc[str(dummy_yr)].index.strftime('%m-%d')

        if self.config.window_years == 'all':
            m_d = self.x_train_pdf.index
        else:
            dummy_yr = np.unique([date.year for date in self.x_train_pdf.index])[
                1]  # Not using index 0 as it might not have full length year
            m_d = self.x_train_pdf.loc[str(dummy_yr)].index.strftime('%m-%d')

        corrected_all = pd.DataFrame()

        # Save original x values for residual analysis
        self.valsx_list = []  # x

        for t in range(len(x_test_pdf)):

            # Get PDF for t
            test_pdf_ = x_test_pdf.iloc[t]

            # Get indices of bias value we're using
            test_pdf_md = test_pdf_.name if self.config.window_years == 'all' else test_pdf_.name.strftime(
                '%m-%d')
            mean_bias_ind = int(np.where(m_d == test_pdf_md)[0])

            # Convert to dataframe & sort
            test_pdf = pd.DataFrame(columns=test_pdf_.keys())
            for i in range(len(test_pdf_)):
                test_pdf.iloc[:,i] = test_pdf_[i]
            # test_pdf = pd.DataFrame(
            #     data={'data': test_pdf_['data'], 'date_raw': test_pdf_['date_raw']})

            sorted_pdf = test_pdf.sort_values('data')
            sorted_dates = sorted_pdf.date_raw
            # order = np.array(sorted_pdf.index) # order index - not used unless comparing day-to-day accuracy

            # Get percentiles info
            arr_x_test = test_pdf['data']  # one pdf
            perc_x, vals_x = self.percentiles(
                arr_x_test, n_quantiles=None)  # quantiles = None here
            self.valsx_list.append(vals_x)  # For unwinding later

            # Split test set into groups (bias value assigned for each quantile segment)
            split_points = np.concatenate(
                ([0], [(perc_train[i] + perc_train[i + 1]) / 2 for i in range(len(perc_train) - 1)], [100]))
            groups = [vals_x[np.where((perc_x > split_points[i]) & (
                perc_x <= split_points[i + 1]))] for i in range(len(split_points) - 1)]

            # Apply correction
            bias = self.mean_bias[mean_bias_ind]
            pred_y = groups * bias if self.config.USE_RATIO else groups + bias
            # Convert to array
            pred_y = np.array(list(itertools.chain.from_iterable(pred_y)))

            # Variance of each prediction
            var_y = np.ones((len(pred_y)))  # DUMMY - use for GP

            corrected = pd.DataFrame(
                data={'date_raw': sorted_dates, 'data': pred_y, 'var': var_y})

            if 'trend_pdf' in sorted_pdf.columns:
                corrected['trend_pdf'] = sorted_pdf.trend_pdf

            corrected_all = corrected_all.append(corrected)

        corrected_all = corrected_all.reset_index(drop=True)

        return corrected_all


    def correct_regression(self,
                           x_test_pdf=None,
                           model=None,
                           ):

        pass

    """
    ================================================================
        Regression models
    ================================================================
    """

    def build_keras_model(self, optimizer, n_nodes, n_nodes2=None, dropout=None, l2_lambda=None, verbose=0):

        pass

    """
    ================================================================
        Static methods
    ================================================================
    """

    @staticmethod
    def percentiles(arr, n_quantiles=4, GET_DF=False, use_quantiles=None):
        """
        Get ordered quantiles from array

        Inputs:
            arr: 1D array
            n_quantiles: number of quantiles to map. Set to None if not interpolating
            GET_DF: get output in pandas dataframe format

         Output:
            Ordered (percentiles, values) if INTERPOLATE=False else output dataframe
        """

        # Pre-sort array and get quantiles for all data
        values = np.array(sorted(arr))
        #nb it is much faster to calculate percentiles en masse this way
        percentiles = rankdata(values) / values.shape[-1] * 100.

        if n_quantiles is not None:
            step = 100 / (n_quantiles)
            percentiles_interp = np.arange(start=step, stop=100, step=step)
            data_interp = np.interp(percentiles_interp, percentiles, values)

            # Update percentiles and values to interpolated
            percentiles = percentiles_interp
            values = data_interp

        if use_quantiles is not None:
            percentiles = use_quantiles
            values = np.percentile(arr, use_quantiles)

        if GET_DF:
            return pd.DataFrame({'percentiles': percentiles, 'data': values})
        else:
            return percentiles, values
