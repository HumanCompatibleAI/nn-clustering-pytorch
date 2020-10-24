import time


def get_random_int_time():
    """
    get a random int based on the least significant part of the time
    """
    time_str = str(time.time())
    dcml_place = time_str.index('.')
    return int(time_str[dcml_place+1:])


def compute_percentile(x, arr):
    r = np.sum(arr < x)
    n = len(arr)
    return r/n
