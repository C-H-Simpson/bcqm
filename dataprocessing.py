"""
========================================================================

Scripts for processing climate data

========================================================================

* How to deal with cftime format

- Github issue:
https://github.com/Unidata/cftime/issues/111

- Use nc_time_axis:
https://github.com/SciTools/nc-time-axis/pull/42

- xarray datetime doc:
http://xarray.pydata.org/en/stable/time-series.html#non-standard-calendars-and-dates-outside-the-timestamp-valid-range

- Cftime doc:
http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#time-coordinate

More:
- Make all these separate sections into different classes?

"""


import numpy as np
import xarray as xr
from . import helper
from .helper import find_nearest, tic, toc
import pandas as pd
import calendar
import itertools


"""
------------------------------------------------------------------------
Data conversion
------------------------------------------------------------------------
"""


def df_to_da(df):
    """ Pandas DataFrame to Xarray data array """
    return df.to_xarray().to_array().squeeze()


def da_to_df(da):
    """ Xarray data array to Pandas DataFrame"""
    da = da.to_pandas().reset_index().set_index('time')
    try:
        da.set_axis(['data'], axis='columns', inplace=True)
    except:
        pass
    return da


"""
------------------------------------------------------------------------
Data extraction
------------------------------------------------------------------------
"""


def load_era(var, year=None, STANDARDISE_COORDS=True, NORMALISE_TO_DATE_ONLY=True, ROLL_LON=False):

    """
    Get ERA-Interim data array (xr)
    Var list = ['T2', 't2max', 't2min']
    Name in array = ['T2', 'MX2T', 'MN2T'], just call by da[da.name]

    # 0: surface tempterature, T2_1979.nc, 1979-2016
    # 1: max temperature, t2max_1979.nc, 1979-2017
    # 2: min temperature, t2min_1979.nc, 1979-2017
    """

    tic()  # Measure time elapsed to load
    # path_parent = '/group_workspaces/jasmin4/bas_climate/data/ecmwf/era-interim/day/'
    path_parent = '/gws/nopw/j04/bas_climate/data/ecmwf1/era-interim/day/'
    # path_parent = '/group_workspaces/jasmin4/bas_climate/data/ecmwf/era5'

    if year is None:  # Get all years

        path = path_parent + var + '/*.nc'
        # Rounding coordinates for ERA-Interim as different years have slightly different coordinate values below 2 decimals
        # https://github.com/pydata/xarray/blob/aabda43c54c6336dc62cc9ec3c2050c9c656cff6/xarray/backends/api.py
        ds = xr.open_mfdataset(path, preprocess=round_coords)

    else:
        # Get specific year
        fname = path_parent + var + '/' + var + '_' + str(year) + '.nc'
        ds = xr.open_dataset(fname)

    da = ds[ds.name]   # Extract main data array from dataset
    da = standardise_da_coords(da, STANDARDISE_COORDS=STANDARDISE_COORDS, NORMALISE_TO_DATE_ONLY=NORMALISE_TO_DATE_ONLY, ROLL_LON=ROLL_LON)
    toc()

    return da.squeeze()


def extract_window(da, lon, lat, lon_window=0, lat_window=0, time=None, INDEX_SLICING=True, ROLL_LON=False):

    """
    Exracts window centering nearest gridpoint to specified lon & lat
    Will roll da if lon or lon-window is -ve (extracting over Greenwich meridian) or if ROLL_LON is True
    Will reorder latitude to go from -ve to +ve

    Inputs:
        da: data array
        lon, lat: centre location
        lon_window, lat_window: position to extract +- window
        time: time snapshot e.g. '1970-01-10'
        INDEX_SLICING: if true, window is the number of adjacent grids]
    """

    da_sliced = da.copy()
    if time is not None:
        da_sliced = da_sliced.sel(time=time)
    da_sliced = ascend_coords(da_sliced)

    if lon - lon_window < 0 or lon < 0:
        print("Rolling longitude 0 to 360 --> -180 to 180")
        da_sliced = roll_lon(da_sliced, CENTRE_GREENWICH=True)

    lon_idx, lon_nearest = find_nearest(da_sliced.lon.values, lon)
    lat_idx, lat_nearest = find_nearest(da_sliced.lat.values, lat)

    # Slice
    if INDEX_SLICING:
        da_sliced = da_sliced.isel(lon=slice(lon_idx - lon_window, lon_idx + lon_window + 1),
                                   lat=slice(lat_idx - lat_window, lat_idx + lat_window + 1))

    else:
        lon_max_idx, lon_max = find_nearest(da_sliced.lon.values, lon + lon_window)
        lon_min_idx, lon_min = find_nearest(da_sliced.lon.values, lon - lon_window)

        lat_max_idx, lat_max = find_nearest(da_sliced.lat.values, lat + lat_window)
        lat_min_idx, lat_min = find_nearest(da_sliced.lat.values, lat - lat_window)

        da_sliced = da_sliced.isel(lon=slice(lon_min_idx, lon_max_idx),
                                   lat=slice(lat_min_idx, lat_max_idx))

    return da_sliced.squeeze()


def slice_time(da, year_min, year_max):
    """
    Slice time by year (inclusive)
    Attempts to fix cases where da.sel(time=slice()) doesn't work because of cftime

    Inputs
        da: Xarray Data Array
        year_min: start year
        year_max: end year
    """

    if type(da.time.values[0]) == np.datetime64:
        try:
            return da.sel(time=slice(str(year_min), str(year_max)))
        except:
            print("Warning: standard indexing of np.datetime64 failed, trying different approach.")
            years = np.array([date.year for date in pd.DatetimeIndex(da.time.values)])
            cond = np.where((years <= year_max) & (years >= year_min))[0]
            return da.isel(time=cond)

    else:
        try:
            years = np.array([date.year for date in da.time.values])
            cond = np.where((years <= year_max) & (years >= year_min))[0]

            return da.isel(time=cond)

        except:
            years = np.array([date.year for date in da.time.values])
            months = np.array([date.month for date in da.time.values])
            days = np.array([date.day for date in da.time.values])

            y = np.isin(years, str(year_min))
            m = np.isin(months, 1)
            d = np.isin(days, 1)
            ind_start = np.argmax(y & m & d)

            y = np.isin(years, str(year_max))
            m = np.isin(months, 12)
            d = np.isin(days, np.max(days))
            ind_end = np.argmax(y & m & d)

            return da.isel(time=slice(ind_start, ind_end + 1))


"""
------------------------------------------------------------------------
Data standardisation all-in-one
------------------------------------------------------------------------
"""


def standardise_da_coords(da, STANDARDISE_COORDS=True, NORMALISE_TO_DATE_ONLY=True, ROLL_LON=False):

    """
     Options
    ---------

    * STANDARDISE_COORDS
    Standardise coordinate names to match GCM convention
    t --> time
    longitude --> lon
    latitude --> lat

    * NORMALISE_TO_DATE_ONLY
    Reset timestamp and keep y-m-d only

    * ROLL_LON
    Roll longitude 0 to +360 --> -180 to +180

    """

    print("Standard initial preprocessing: \n")

    if STANDARDISE_COORDS:
        print("Standardising coordinate names, ordering and rounding to 2 dp...")
        da = rename_coord(da, 't', 'time')
        da = rename_coord(da, 'longitude', 'lon')
        da = rename_coord(da, 'latitude', 'lat')

        if 'lat' in da.coords.dims  and 'lon' in da.coords.dims: # to accept 1D DataArrays
            if da.lat.values.size > 1 and da.lon.values.size > 1:
                da = ascend_coords(da)  # Reorder coordinates to ascend
            da = round_coords(da)   # Round coordinates to 2 decimals
    # else:
    #     print("(Not standardising coordinates)")

    if NORMALISE_TO_DATE_ONLY:
        print("Converting datetime to y-m-d (reset timestamp)...")
        da = normalise_to_date_only(da)
    # else:
    #     print("(Not standardising datetime)")

    if ROLL_LON:
        print("Rolling longitude...")
        da = roll_lon(da)
    # else:
    #     print("(Not rolling longitude)")

    print("\nStandard initial preprocessing done. \n")

    return da


def simple_preprocess_1d(da,
                         lon=None,
                         lat=None,
                         year_start=None,
                         year_end=None,
                         GET_DF=True,
                         NO_LEAP=True,
                         interp_method='nearest',
                         verbose=True,
                         ):
    """
    Slice by longitude and latitude (1D) and years and get pandas df
    """

    # (1) Standardise coordinates (default setting)
    da = standardise_da_coords(da)

    # (2) Roll longitude if necessary and extract coordinates
    if (lon is None) & (lat is None) & (da.ndim is 1):
        # if da is 1D, and lon and lat are not specified, then do not attempt to interpolate grid
        # i.e., simply output data for a timeseries using current coordinate system
        pass
    else:
        if da.lon.values.size > 1 or da.lat.values.size > 1:
            if lon is not None and lat is not None:
                if lon < 0:
                    da = roll_lon(da)
                print("Extracting coordinate...")
                da = da.interp(coords={'lon': lon, 'lat': lat}, method=interp_method)
            else:
                raise ValueError("Data array given not 1D - please specify longitude and latitude")

    # (3) Slice time
    if year_start is not None and year_end is not None:
        print("Slicing time...")
        da = slice_time(da, year_start, year_end)

    if year_start is not None and year_end is None:
        da = slice_time(da, year_start, pd.to_datetime(da.time.values[-1]).year)

    if year_start is None and year_end is not None:
        da = slice_time(da, pd.to_datetime(da.time.values[0]).year, year_end)

    # (4) Standardise datetime format
    da = convert_to_datetime64(da, verbose)

    if GET_DF:
        # (5) Convert into dataframe
        print("Converting to dataframe...")
        df = da_to_df(da)

    if NO_LEAP and GET_DF:
        print("Removing 29-Feb's...")
        # (6) Eliminate 29th Feb in leap years so that each year is 365 days
        df = remove_leapyears(df)

    print("Done simple preprocessing, ouput 1D dataframe. \n")

    return df if GET_DF else da


"""
------------------------------------------------------------------------
Data cleaning - longitude/latitude coordinates
------------------------------------------------------------------------
"""


def roll_lon(da, CENTRE_GREENWICH=True):
    """
    Roll climate data longitude 0 to 360 --> -180 to 180
    ! Doesn't check if already rolled

    https://gis.stackexchange.com/questions/201789/verifying-formula-that-will-convert-longitude-0-360-to-180-to-180
    """
    print("Rolling longitude 0 to 360 --> -180 to 180...")

    if CENTRE_GREENWICH:
        da_E = da.where(da.lon <= 180, drop=True)   # Split E and W
        da_W = da.where(da.lon > 180, drop=True)
        da_W['lon'] = (da_W.lon + 180) % 360 - 180  # Change W coordinate -180 --> 0

    else:
        # lon3=mod(lon1,360)
        da_W = da.where(da.lon <= 0, drop=True)   # Split E and W
        da_E = da.where(da.lon > 0, drop=True)
        da_W['lon'] = da_W.lon % 360              # Change W coordinate 0 --> 360

    da = xr.concat([da_W, da_E], dim='lon')
    da = da.chunk({'lon': da.sizes['lon']})       # Retain longitude chunk size (required for xr interpolation)

    print("Done")
    return da   # Combine W and E


def rename_coord(da, old_name, new_name):
    """ Rename coordinate in xr data array """
    if old_name in list(da.coords.keys()):
        da = da.rename({old_name:new_name})

    return da


def ascend_coords(da):
    """
    Sort latitude and longitude in ascending order
    (Useful for index selection to extract neighbouring pixels)
    """

    try:
        if da.lat[0].values > da.lat[-1].values:
            da = da.reindex(lat=da.lat[::-1])

        if da.lon[0].values > da.lon[-1].values:
            da = da.reindex(lon=da.lon[::-1])
    except:
        if da.latitude[0].values > da.latitude[-1].values:
            da = da.reindex(latitude=da.latitude[::-1])

        if da.longitude[0].values > da.longitude[-1].values:
            da = da.reindex(longitude=da.longitude[::-1])

    return da


def round_coords(da, decimals=2):
    """
    Round up long + lat coordinates
    (Useful for aligning ERA-Interim data where different years have slightly different coodinate values but use with care)
    TO DO: use np.isclose instead 
            see https://github.com/scott-hosking/baspy/blob/4640eeec81ecfd0c319fb915916aaab4d0717167/_iris/util.py#L203
    """
    try:
        da['lat'] = np.around(da.lat, decimals=decimals)
        da['lon'] = np.around(da.lon, decimals=decimals)
    except:
        da['latitude'] = np.around(da.latitude, decimals=decimals)
        da['longitude'] = np.around(da.longitude, decimals=decimals)

    return da


"""
------------------------------------------------------------------------
Data cleaning - datetime
------------------------------------------------------------------------
"""


def normalise_to_date_only(da):
    """
    Change time coordinate to date only
    Useful for combining xarrays where time is inconsistent e.g. tas and tasmin/tasmax in ERA-Interim
    ! Only works with datetime64 format
    """

    da = rename_coord(da, 't', 'time')

    try:
        da['time'] = da.indexes['time'].normalize()
    except:
        try:
            if type(da.time.values[0]) != np.datetime64:
                da['time'] = da.indexes['time'].to_datetimeindex().normalize()
        except:
            print("Warning: Failed. Either existing format incompatible or \n \
                  'time' coordinate does not exist. \n \
                  Ignore warning if converting/interpolating to datetime64 later.")

    return da


def convert_to_datetime64(da, verbose=True):
    """
    ! ONLY WORKS FOR 1D DATA (i.e. ONE GRID LOCATION)
    Convert datetime format to standard python datetime by simple interpolation
    """

    if type(da.time.values[0]) == np.datetime64:
        print("Already in datetime64 format.")
        return da

    print("Converting to datetime64... (retrieving all values and interpolating to fit standard format if necessary - may take a while)")

    start_year = da.time.values[0].year
    end_year = da.time.values[-1].year

    # Create new data array
    times = np.arange(str(start_year), str(end_year + 1), dtype='datetime64[D]')
    new_da = xr.DataArray(np.zeros(len(times)), coords=[('time', times)])

    for yr in np.arange(start_year, end_year + 1):

        if len(da.sel(time=str(yr))) != len(new_da.sel(time=str(yr))):

            # Do interpolation of original array to fit standard datetime
            if verbose:
                print("    Processing year " + str(yr) + "/" + str(end_year))
                tic()
            leapyear = calendar.isleap(yr)
            size = 365 if not leapyear else 366
            arr = da.sel(time=str(yr)).values
            arr_new = helper.interp1d(arr, size)
            if verbose:
                toc()

        else:
            arr_new = da.sel(time=str(yr)).values

        # Replace array values with new data
        ind_start = (new_da.indexes['time'] == pd.Timestamp(str(yr) + '-01-01')).argmax()
        ind_end = (new_da.indexes['time'] == pd.Timestamp(str(yr) + '-12-31')).argmax()
        new_da[ind_start:ind_end + 1] = arr_new

    print("Done")
    return new_da


def remove_leapyears(df):
    is_leap_and_29Feb = (df.index.year % 4 == 0) & ((df.index.year % 100 != 0) | (df.index.year % 400 == 0)) & (df.index.month == 2) & (df.index.day == 29)
    return df.loc[~is_leap_and_29Feb]


"""
------------------------------------------------------------------------
Subsample dataframes
------------------------------------------------------------------------
"""


def subsample_days(df, n_subsample_days=7):
    """
    Subsample PDFs every n_subsample days (neighbouring PDFs are very similar)
    Same m-d used for every year so biases can be averaged for all training data
    e.g. df_h_filtered = subsample_days(df_h, 15)
    """
    dates_all = df.loc[str(df.index.year[0])].index
    dates_to_use = dates_all[np.arange(int(np.floor(n_subsample_days / 2)), len(dates_all), n_subsample_days)].strftime('%m-%d')
    df_ = df[df.index.strftime('%m-%d').isin(dates_to_use)].copy()
    return df_


def subsample_years(df, n_subsample_years=3):
    """
    Subsample PDFs every n_subsample years
    """
    years_all = np.unique([date.year for date in df.index])
    inds = np.arange(0, len(years_all), n_subsample_years)
    years_sub = years_all[inds]
    df_ = df[df.index.year.isin(years_sub)].copy()
    return df_


def subsample(df, n_subsample_days=7, n_subsample_years=3):
    """
    Subsample pdf: days and years combined
    """

    if len(df) <= 365:
        # If window_years = 'all' then pdf will be 365 long
        df_ = df.iloc[np.arange(0, len(df), n_subsample_days)]

    else:
        df_ = subsample_days(df, n_subsample_days)
        df_ = subsample_years(df_, n_subsample_years)

    return df_


"""
------------------------------------------------------------------------
Other functions to manage dataframes
------------------------------------------------------------------------
"""

def intersect(x, y):
    """
    Return filtered x and y dataframes with intersecting dates
    """
    intersecting_dates = x.index.intersection(y.index)
    x_ = x.loc[intersecting_dates]
    y_ = y.loc[intersecting_dates]
    return x_, y_


"""
------------------------------------------------------------------------
Processing for regression predictors
------------------------------------------------------------------------
"""

def convert_circular(x, cycle_length):
    """ Convert value or array to circular variable """
    return np.round(np.sin((2 * np.pi * (x + 0.5)) / cycle_length), 3)


def normalise_data(X, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    X = (X - mean) / std
    return X, mean, std


"""
------------------------------------------------------------------------
CLASS: Timseries dataframe into another dataframe
------------------------------------------------------------------------
"""

class Df_to_df:

    def __init__(self):
        pass

    def detrend_pdf(self,
                    df_pdf,
                    trend,
                    #SAVE_RAW_TREND=False,
                    verbose=True,
                    ):
        """
        Output PDF detrended with extra column of raw trend if SAVE_RAW_TREND is True
        """

        if verbose:
            print("Detrending pdf...")
        trend_, _ = intersect(trend, df_pdf)
        df_pdf_detrended = df_pdf.copy()
        df_pdf_detrended['data'] = (df_pdf[['data']] - trend_).values

        assert len(df_pdf_detrended) == len(df_pdf)
        if verbose:
            print("Done")

        return df_pdf_detrended


    def create_pdf(self, df,
                   window_days=31,
                   window_years='all',
                   n_subsample_days=None,
                   PRESERVE_DATES=True,
                   PRESERVE_EDGES=True,
                   detrend_trend=None, ### Trend to subtract from the pdf
                   trend_pdf=None,
                   add_trend_pdf=None,
                   verbose=True,
                   ):

        """
        Create dataframe of PDF values with day and year window + subsampling
        """
        if verbose:
            print("Creating pdf...")

        PRESERVE_EDGES = True if window_years is 'all' else PRESERVE_EDGES

        # ---------------- #
        #  Roll windows    #
        # ---------------- #
        df_ = self.rolling_list(df,
                                window_days=window_days,
                                window_years=window_years,
                                PRESERVE_DATES=PRESERVE_DATES,
                                PRESERVE_EDGES=PRESERVE_EDGES,
                                detrend_trend=detrend_trend, ### Trend to subtract from the pdf
                                trend_pdf=trend_pdf,
                                add_trend_pdf=add_trend_pdf,
                                verbose=verbose,
                                )

        # ------------- #
        #  Subsample    #
        # ------------- #
        # if n_subsample_days is not None or n_subsample_days is not 0:
        if n_subsample_days is not None and n_subsample_days is not 0:
            if verbose:
                print("Subsampling pdf...")
            df_ = subsample(df_, n_subsample_days, 1)

        if verbose:
            print("Done: pdf created.")

        return df_


    def rolling_list(self, df,
                     window_days=31,
                     window_years=7,
                     PRESERVE_DATES=False,
                     PRESERVE_EDGES=True,
                     MATCH_LENGTH_IN_YR=False,
                     detrend_trend=None, ### Trend to subtract from the pdf
                     trend_pdf=None,
                     add_trend_pdf=None,
                     verbose=True,
                     ):

        """
        Returns dataframe with list of values in specified window.
        If window_years larger than maximum allowed or set to 'all', uses all available years in df.

        PRESERVE_EDGES - even if using window_days the edges are preserved. Not done for window_years

        detrend_trend - trend to use for detrending pdf
        """

        # ----------------------------- #
        #  Checking and preprocessing   #
        # ----------------------------- #
        # Make sure that we are only working with 'data' column (in case there is a 'var' column)
        df = df[['data']]

        SAVE_PDF_TREND = True if trend_pdf is not None else False

        if (window_days % 2) == 0:
            raise ValueError('window_days must be odd.')

        if window_years != 'all':
            if (window_years % 2) == 0:
                raise ValueError('window_years must be odd.')

        max_window_years = df.index[-1].year - df.index[0].year
        if window_years is 'all':
            window_years_original = 'all'
            window_years = max_window_years + 1

        # ---------------------- #
        #  Rolling days window   #
        # ---------------------- #

        dates = []
        vals = []
        date_raw = []

        if PRESERVE_EDGES:
            raw_vals = df.values.squeeze()
            dates = df.index.to_list()
            for i in range(len(df)):
                i_s = i - int(np.floor(window_days / 2))
                i_s = 0 if i_s < 0 else i_s

                i_e = i + int(np.floor(window_days / 2))
                i_e = -1 if i_e >= len(df) else i_e
                ii_e = None if i_e == -1 else i_e+1

                #vals.append(df.loc[str(start_day) : str(end_day)].values.squeeze())
                #vals.append(df.iloc[i_s : ii_e].values.squeeze())
                vals.append(raw_vals[i_s : ii_e])

                if PRESERVE_DATES:
                    date_raw.append(df.index[i_s : ii_e])

        else:
            # for start in range(len(df) - window_days):
            for start in range(len(df) - window_days + 1):

                start_day = df.index[start]
                end_day = df.index[start + window_days - 1]
                mid_day = df.index[int(start + window_days / 2)]

                i_s = start
                ii_e = start + window_days
                #vals.append(df.loc[str(start_day) : str(end_day)].values.squeeze())
                vals.append(df.iloc[i_s : ii_e].values.squeeze())
                dates.append(mid_day)

                if PRESERVE_DATES:
                    date_raw.append(df.index[i_s : ii_e])

        df_ = pd.DataFrame(index=dates, data={'data':vals})

        # ---------------------------------------- #
        #  Detrend PDF here if MODE is detrend_pdf #
        # ---------------------------------------- #
        if detrend_trend is not None:
            df_ = self.detrend_pdf(df_,
                                   detrend_trend,
                                   verbose=verbose,
                                   )

        if SAVE_PDF_TREND:
            # df_['pdf_date'] = df_.index
            if verbose:
                print("Saving pdf trends... (warn: only works if grouping all years)")

            def tile_trend(df_line):
                return np.repeat(df_line['trend_pdf'], len(df_line['data']))

            df_['trend_pdf'] = trend_pdf.values
            df_['trend_pdf'] = df_.apply(tile_trend, axis=1)

            if verbose:
                print("Done")

        if PRESERVE_DATES:
            df_['date_raw'] = date_raw

        # ----------------------- #
        #  Rolling years window   #
        # ----------------------- #

        if window_years == 1:
            """ Do not roll years """
            pass

        elif window_years > 1 and window_years <= max_window_years:

            df = df_

            df = remove_leapyears(df)

            # All beginning years
            years_list = np.unique([date.year for date in dates])
            beginning_years_len = int(len(years_list) - np.floor(window_years / 2) * 2)
            years = years_list[:beginning_years_len]
            midyears_ = years_list[int(window_years / 2) : -int(window_years / 2)]

            vals_ = []
            date_raw_ = []

            for i, yr in enumerate(years):

                start_year = yr
                end_year = years_list[i + window_years - 1]

                df_slice = df.loc[str(start_year) : str(end_year)]
                group = df_slice.groupby([df_slice.index.month, df_slice.index.day])
                vals = group['data'].apply(list)
                vals = [np.concatenate(i) for i in vals]
                vals_.append(vals)

                if PRESERVE_DATES:
                    date_raw = group['date_raw'].apply(list)
                    date_raw = [np.concatenate(i) for i in date_raw]
                    date_raw_.append(date_raw)

            index = pd.date_range(str(midyears_[0]) + '-01-01', str(midyears_[-1]) + '-12-31')
            index = index[index.year.isin(midyears_)]

            if PRESERVE_DATES:
                df_ = pd.DataFrame(index=index, columns=['data', 'date_raw'])
                df_ = remove_leapyears(df_)

                df_['data'] = np.array(list(itertools.chain.from_iterable(vals_)))
                df_['date_raw'] = np.array(list(itertools.chain.from_iterable(date_raw_)))

            else:
                df_ = pd.DataFrame(index=index, columns=['data'])
                df_ = remove_leapyears(df_)
                df_['data'] = np.reshape(vals_, np.shape(vals_)[0] * np.shape(vals_)[1])

        elif window_years > max_window_years:

            if window_years_original is not 'all':
                print("WARNING: window_years is greater than len(years) in data, grouping all")
            df_ = self.group_years_in_list(df_,
                                           PRESERVE_DATES,
                                           #SAVE_RAW_TREND,
                                           SAVE_PDF_TREND,
                                           )

        else:
            raise ValueError('Invalid window_years')

        if MATCH_LENGTH_IN_YR:
            # Check length (len of unique m-d) is the same in all years
            # (May not be if some years are incomplete in original data)
            # If not, only select years with majority length
            # This was (probably) only necessary for old code using "time_of_year_idx"

            years_all = np.unique([date.year for date in df_.index])
            lens = [len(df_.loc[str(yr)].index) for yr in years_all]

            if len(set(lens)) != 1:
                # Mask majority
                print("WARNING: length (len of unique m-d) is not the same in all years, masking majority length years")
                mask = np.isin(lens, max(set(lens), key=lens.count))
                selected_years = years_all[mask]
                df_ = df_[df_.index.year.isin(selected_years)]

        return df_


    @staticmethod
    def group_years_in_list(df,
                            PRESERVE_DATES=False,
                            SAVE_PDF_TREND=False):
        """
        Group years and give one value for each day and month
        """

        grouped = df.groupby([df.index.month, df.index.day])
        df_data = grouped['data'].apply(np.array)
        vals = [np.concatenate(i) for i in df_data]
        dummy_yr = np.unique([date.year for date in df.index])[1]
        df_ = pd.DataFrame(index=df.loc[str(dummy_yr)].index.strftime('%m-%d'), data={'data':vals})

        if PRESERVE_DATES:
            df_date_raw = grouped['date_raw'].apply(np.array)
            date_raw = [np.concatenate(i) for i in df_date_raw]
            df_['date_raw'] = date_raw

        if SAVE_PDF_TREND:
            df_trend_pdf = grouped['trend_pdf'].apply(np.array)
            trend_pdf = [np.concatenate(i) for i in df_trend_pdf]
            df_['trend_pdf'] = trend_pdf

        return df_


    @staticmethod
    def shuffle_rows(df):
        return df.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def unnesting(df, columns_to_explode):
        """
        df.explode() for multiple columns
        """
        idx = df.index.repeat(df[columns_to_explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in columns_to_explode], axis=1)
        df1.index = idx

        return df1.join(df.drop(columns_to_explode, 1), how='left')


"""
------------------------------------------------------------------------
Detrend timeseries data
------------------------------------------------------------------------
"""

class Detrend:

    def __init__(self, MODE=None, polyfit_degrees=2):
        self.polyfit_degrees = polyfit_degrees
        self.MODE = MODE
        self.df_to_df = Df_to_df()


    def detrend_by_pdf(self, df, df_pdf, detrend_using=np.median):

        dummy_yr = np.unique([date.year for date in df.index])[1]  # Not using index 0 as it might not have full length year
        days = df.loc[str(dummy_yr)].index.strftime('%m-%d')

        df_trend_list = []

        pdf_m = pd.DataFrame(df_pdf['data'].apply(detrend_using))

        for i, m_d in enumerate(days):  # Process specific month and day

            month = m_d[:2]
            day = m_d[-2:]

            df_ = pdf_m.loc[(pdf_m.index.month == int(month)) & (pdf_m.index.day == int(day))]
            df_trend = self.get_trend(df_, polyfit_degrees=self.polyfit_degrees, GET_DF=True)

            df_trend_list.append(df_trend)

        df_trend = pd.concat(df_trend_list).sort_index()

        assert len(df_trend) == len(df)

        self.df_summary = pdf_m

        return df, df_trend  # df is original


    def detrend_by_md(self, df, df_pdf=None):
        """
        We need df_pdf if there are multiple values per y-m-d (e.g. want to represent y-m-d values with ±x days (or per season))
        """

        dummy_yr = np.unique([date.year for date in df.index])[1]  # Not using index 0 as it might not have full length year
        days = df.loc[str(dummy_yr)].index.strftime('%m-%d')

        df_detrended_list = []
        df_trend_list = []

        for i, m_d in enumerate(days):

            # Process specific month and day
            month = m_d[:2]
            day = m_d[-2:]

            df_ = df.loc[(df.index.month == int(month)) & (df.index.day == int(day))]

            if self.MODE is 'detrended_md_roll':

                """ If there are multiple values per m-d pair """

                if df_pdf is None:
                    raise ValueError("Please specify df_pdf in argument")

                # Get pdf version of df
                df_pdf_ = df_pdf.loc[(df_pdf.index.month == int(month)) & (df_pdf.index.day == int(day))]

                # Explode df - unpack list into separate rows
                rows = []
                _ = df_pdf_.apply(lambda row: [rows.append([row.name, nn]) for nn in row.values[0]], axis=1)
                df_exploded = pd.DataFrame(rows, columns=['time', 'data']).set_index('time')

                # Now get detrended timeseries (1) get trend (2) subtract this from df
                df_lookup = pd.DataFrame(data={'time': df_pdf_.index, 'idx': np.arange(len(df_pdf_.index))})
                df_merged = df_exploded.copy()
                df_merged.reset_index(level=0, inplace=True)
                df_merged = df_merged.merge(df_lookup)

                # Fit polynomial and get trend (put this in function)
                x_ = df_merged.idx.values
                y_ = df_merged.data.values
                z_ = np.polyfit(x_, y_, self.polyfit_degrees)
                f_ = np.poly1d(z_)
                trend = f_(np.arange(len(df_)))
                df_trend = pd.DataFrame(index=df_.index, data={'data': trend})

                # Detrend
                detrended_timeseries = df_.values.squeeze() - df_trend['data'].values.squeeze()
                df_detrended = pd.DataFrame(index=df_.index, data={'data': detrended_timeseries})

            elif self.MODE is 'detrended_md':
                """ If there is only one value per m-d pair """
                df_detrended, df_trend = self.detrend_df(df_, self.polyfit_degrees)

            elif self.MODE is 'subtract_median_md':
                df_detrended, df_trend = self.subtract_median_md(df_)

            elif self.MODE is None:
                print("MODE is None in Detrend item, please set valid MODE")

            df_detrended_list.append(df_detrended)
            df_trend_list.append(df_trend)

        df_detrended = pd.concat(df_detrended_list).sort_index()
        df_trend = pd.concat(df_trend_list).sort_index()

        assert len(df_detrended) == len(df)
        assert len(df_trend) == len(df)

        return df_detrended, df_trend


    def detrend_df(self, df, polyfit_degrees=None, df_trend=None):
        """
        Return detrended dataframe with trend info
        Df must have index as dates which is converted into
        index values [0, 1, 2, ...] when fitting polynomial
        """
        polyfit_degrees = self.polyfit_degrees if polyfit_degrees is None else polyfit_degrees

        df_trend = self.get_trend(df, polyfit_degrees, GET_DF=True) if df_trend is None else df_trend
        detrended_timeseries = df.values.squeeze() - df_trend['data'].values.squeeze()
        df_ = pd.DataFrame(index=df.index, data={'data': detrended_timeseries})

        return df_, df_trend


    def get_trend(self, df, polyfit_degrees=None, GET_DF=True):

        polyfit_degrees = self.polyfit_degrees if polyfit_degrees is None else polyfit_degrees

        x_ = np.arange(len(df.values))
        y_ = df.values.squeeze()

        z_ = np.polyfit(x_, y_, polyfit_degrees)
        f_ = np.poly1d(z_)
        trend = f_(x_)

        if GET_DF:
            df_trend = pd.DataFrame(index=df.index, data={'data': trend})
            return df_trend
        else:
            return trend


    def subtract_median_md(self, df, window_days=31):

        # df_to_df = Df_to_df()
        df_pdf = self.df_to_df.create_pdf(df, window_days=window_days, window_years='all', PRESERVE_DATES=False, PRESERVE_EDGES=False) # Should do preserve edges here?
        df_m = df_pdf.data.apply(np.median)
        df_m_tiled = np.tile(df_m, int(len(df) / 365))
        assert len(df_m_tiled) == len(df)

        df_ = df.copy()
        df_['data'] = df.values.squeeze() - df_m_tiled
        df_median_ = df.copy()
        df_median_['data'] = df_m_tiled

        return df_, df_median_
