import time
import numpy as np
import datetime
import pickle
import scipy.interpolate as interp
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from scipy.signal import savgol_filter


"""
------------------------------------------------------------------------
Dictionary of areas
------------------------------------------------------------------------
"""

class Dicts:

    years = defaultdict(lambda: defaultdict(int))
    years['training'] = {'start': 1979, 'end': 2004}
    years['projecting'] = {'start': 2006, 'end': 2100}

    city = defaultdict(lambda: defaultdict())
    city['london'] = {'lat': 51.50735, 'lon': -0.127758}
    city['madrid'] = {'lat': 40.4168, 'lon': -3.7038}
    city['paris'] = {'lat': 48.8566, 'lon': 2.3522}
    city['cairo'] = {'lat': 30.0444, 'lon': 31.2357}
    city['tokyo'] = {'lat': 35.6895, 'lon': 139.6917}
    city['nyc'] = {'lat': 40.7128, 'lon': -74.0060}
    city['beijing'] = {'lat': 39.9042, 'lon': 116.4074}
    city['dublin'] = {'lat': 53.3498, 'lon': -6.2603}

    continent = defaultdict(lambda: defaultdict())
    continent['europe'] = {'lon_bnds':(-11.25, 33.75), 'lat_bnds':(35.1, 72.5)}
    continent['europe_atlantic'] = {'lon_bnds':(-50., 33.75),   'lat_bnds':(25.0, 72.5)}

    country = defaultdict(lambda: defaultdict())
    country['uk'] = {'lon_bnds':(-11, 2), 'lat_bnds':(48, 60)}
    country['france'] = {'lon_bnds':(-5, 9),  'lat_bnds':(41, 52)}
    country['spain'] = {'lon_bnds':(-11, 6), 'lat_bnds':(35, 45)}
    country['egypt'] = {'lon_bnds':(22, 39), 'lat_bnds':(20, 34)}
    country['china'] = {'lon_bnds':(72,135), 'lat_bnds':(20,55)}


"""
------------------------------------------------------------------------
Get clean date-time stamp
------------------------------------------------------------------------
"""


def stamp():
    """ Get clean date-time stamp """
    dt = str(datetime.datetime.now()).split('.')[0]
    dt = dt.replace(':', '-')
    dt = dt.replace(' ', '-')
    return dt


"""
------------------------------------------------------------------------
Pickle save and restore - 'name' to include path
------------------------------------------------------------------------
"""


def save_pickle(data, name):
    with open(name, "wb") as fp:   # Unpickling
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved at " + name)


def open_pickle(name):
    with open(name, "rb") as fp:   # Unpickling
        return pickle.load(fp)


"""
------------------------------------------------------------------------
Generator for elaped time to run a code block
------------------------------------------------------------------------
"""


def TicTocGenerator():
    ti = 0           # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()


def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


"""
------------------------------------------------------------------------
Smoothing function (moving average) for plotting
------------------------------------------------------------------------
https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

Better:
from scipy.signal import savgol_filter
yhat = savgol_filter(y, 51, 3)  # window size 51, polynomial order 3
"""


def smooth(array, smoothing_horizon=100., initial_value=0.):
    """
    Not a good smoothing method, shifts smoothed values to the right - only use for plot aesthetic purposes!
    """
    smoothed_array = []
    value = initial_value
    b = 1. / smoothing_horizon
    m = 1.
    for x in array:
        m *= 1. - b
        lr = b / (1 - m)
        value += lr * (x - value)
        smoothed_array.append(value)

    return np.array(smoothed_array)


def rolling_mean_npconv(array, window, mode='valid'):
    """
    Don't use other modes - downweights edges
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    """
    return np.convolve(array, np.ones((window,)) / window, mode=mode)


def smooth_savgol(array, window=201, poly_order=3):
    return savgol_filter(array, window, poly_order)


"""
Seasonal Adjustment
https://stackoverflow.com/questions/47076771/statsmodels-seasonal-decompose-what-is-naive-about-it
https://stackoverflow.com/questions/34494780/time-series-analysis-unevenly-spaced-measures-pandas-statsmodels

"""


"""
------------------------------------------------------------------------
Other functions
------------------------------------------------------------------------
"""


def find_nearest(array, value):
    # Returns index and array value of nearest provided
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def interp1d(arr, size):
    """
    Simple 1D interpolation to resize array
    size: final size you want
    """

    arr_interp = interp.interp1d(np.arange(arr.size), arr)
    return arr_interp(np.linspace(0, arr.size - 1, size))


def split(a, n):
    """
    List splitter
    >>> list(split(range(11), 3))
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# ---------------------------------- #
#  Other methods                     #
# ---------------------------------- #

def sigmoid(x):
    """ Numerically stable sigmoid """
    z = np.exp(x)
    return z / (1 + z)


def crazyshuffle(arr):
    """ Shuffle each column separately """
    x, y = arr.shape
    rows = np.indices((x, y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols].T


# ---------------------------------- #
#  Validation related                #
# ---------------------------------- #

def rmse(y_actual, y_predicted):
    return np.sqrt(mean_squared_error(y_actual, y_predicted))


# ---------------------------------- #
#  Datetime related                  #
# ---------------------------------- #


def is_leap_and_29Feb(df):
    # Returns mask for leapyear 29th Feb
    return (df.index.year % 4 == 0) & ((df.index.year % 100 != 0) | (df.index.year % 400 == 0)) & (df.index.month == 2) & (df.index.day == 29)
