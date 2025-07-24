"""
Numba toolbox with helper functions.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------
This module implements numba compatible versions of standard numpy functions.
Standard numba functionality often supports only limited arguments.

The algorithms were partially used in Gassilloud et al. (2025) [1].
The Claude 3.7 Sonnet large language model (Anthropic, 2024) was used to complete
the function docstrings.


References:

[1] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""


import numpy as np
from numba.core import types
from numba import jit, objmode, prange


#%% elementar functions

@jit(nopython=True, cache=True)
def nb_min(x):
    return np.min(x)


@jit(nopython=True, cache=True)
def nb_max(x):
    return np.max(x)


@jit(nopython=True, cache=True)
def nb_mean(x):
    return np.mean(x)


@jit(nopython=True, cache=True)
def nb_count(x):
    return float(x.shape[0])


@jit(nopython=True, cache=True)
def nb_std(x):
    return np.std(x)



#%% axis any / all / ...

@jit(nopython=True, cache=True)
def nb_any_axis1(x):
    """
    Numba compatible version of np.any(x, axis=1).
    
    This function computes the logical OR of array elements over the second axis (axis=1)
    for a 2D boolean array.
    For further information see: https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
    
    Parameters
    ----------
    x : ndarray
        Input 2D array of boolean values.
        
    Returns
    -------
    ndarray
        1D boolean array with True where at least one element along axis 1 is True,
        False otherwise.
    
    Notes
    -----
    This function provides the same functionality as np.any(x, axis=1) but is
    implemented in a way that is compatible with Numba's JIT compilation.
    """
    out = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@jit(nopython=True, cache=True)
def nb_any_axis0(x):
    """
    Numba compatible version of np.any(x, axis=0).
    
    This function computes the logical OR of array elements over the first axis (axis=0)
    for a 2D boolean array. For further information see:
    https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
    
    Parameters
    ----------
    x : ndarray
        Input 2D array of boolean values.
        
    Returns
    -------
    ndarray
        1D boolean array with True where at least one element along axis 0 is True,
        False otherwise.
    
    Notes
    -----
    This function provides the same functionality as np.any(x, axis=0) but is
    implemented in a way that is compatible with Numba's JIT compilation.
    """
    out = np.zeros(x.shape[1], dtype=np.bool_)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@jit(nopython=True, cache=True)
def np_all_axis1(x):
    """
    Numba compatible version of np.all(x, axis=1).
    
    This function computes the logical AND of array elements over the second axis (axis=1)
    for a 2D boolean array. For further information see:
    https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
    
    
    Parameters
    ----------
    x : ndarray
        Input 2D array of boolean values.
        
    Returns
    -------
    ndarray
        1D boolean array with True for rows where all elements are True,
        False otherwise.
    
    Notes
    -----
    This function provides the same functionality as np.all(x, axis=0) but is
    implemented in a way that is compatible with Numba's JIT compilation.
    """
    out = np.ones(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


@jit(nopython=True, cache=True)
def np_all_axis0(x):
    """
    Numba compatible version of np.all(x, axis=0).
    
    This function computes the logical AND of array elements over the first axis (axis=0)
    for a 2D boolean array. For further information see:
    https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
    
    
    Parameters
    ----------
    x : ndarray
        Input array to test elements along axis 0.
        Must be a 2D array.
    
    Returns
    -------
    ndarray
        Boolean array with the same shape as x.shape[1], where each element
        indicates whether all values in the corresponding column of x are True.
    
    Notes
    -----
    This function provides the same functionality as np.all(x, axis=1) but is
    implemented in a way that is compatible with Numba's JIT compilation.
    """
    out = np.ones(x.shape[1], dtype=np.bool_)
    for i in range(x.shape[0]):
        out = np.logical_and(out, x[i, :])
    return out



#%% apply along axis

@jit(nopython=True, parallel=True)
def _apply_along_axis_0(func1d, arr, out):
    """
    Apply a function along axis 0 of an array in parallel.

    This function applies a 1-D function to each slice of an array along axis 0 
    and stores the results in a pre-allocated output array. It's designed to be 
    used with Numba's parallel capabilities (prange).

    Parameters
    ----------
    func1d : callable
        The function to apply to 1-D slices. Should accept a 1-D array and return a scalar.
    arr : ndarray
        The input array with dimensions >= 2.
    out : ndarray
        The pre-allocated output array to store results. Should have shape matching 
        the dimensions of arr with the first dimension removed.

    Raises
    ------
    RuntimeError
        If the input array has fewer than 2 dimensions.

    Notes
    -----
    - This is a helper function implementing the core functionality for applying functions
      along axis 0 in a parallelized manner.
    - For 2D arrays, the function processes columns in parallel.
    - For higher dimensional arrays, the function processes recursively.
    - Requires Numba's prange to be available in the scope.
    """
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in prange(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@jit(nopython=True)
def apply_along_axis_0(func1d, arr):
    """
    Apply a function along the first axis (axis=0) of an array.

    This function mimics the behavior of `numpy.apply_along_axis` with `axis=0`, 
    but is compatible with Numba JIT compilation.
    For further information see: https://github.com/numba/numba/issues/1269 

    Parameters
    ----------
    func1d : callable
        The function to apply. Must take a 1-D array as its first argument and 
        return a scalar or 1-D array.
    arr : numpy.ndarray
        The input array. Must have at least one dimension and non-zero size.

    Returns
    -------
    numpy.ndarray
        The output array. For an input array with shape (n, m1, m2, ...), the output
        will have shape (m1, m2, ...) if `func1d` returns a scalar, or 
        (k, m1, m2, ...) if `func1d` returns an array of length k.

    Raises
    ------
    RuntimeError
        If `arr` has zero size or is a scalar (zero dimensions).

    Notes
    -----
    This function requires an implementation of `_apply_along_axis_0` which is not shown
    in the provided code.
    """

    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out



#%% elementar functions along axis

@jit(nopython=True)
def nb_mean_axis_0(arr):
    return apply_along_axis_0(np.mean, arr)


@jit(nopython=True)
def nb_median_axis_0(arr):
    return apply_along_axis_0(np.median, arr)


@jit(nopython=True)
def nb_amin_axis_0(arr):
    return apply_along_axis_0(np.amin, arr)


@jit(nopython=True)
def nb_amax_axis_0(arr):
    return apply_along_axis_0(np.amax, arr)


@jit(nopython=True)
def nb_std_axis_0(arr):
    return apply_along_axis_0(np.std, arr)



#%% special functions

@jit(nopython=True)
def nb_unique_axis0(array, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique rows in a 2D array using Numba.
    This function returns the sorted unique rows of a 2D array, optionally with
    indices that can be used to reconstruct the original array.
    Parameters
    ----------
    array : ndarray
        Input 2D array.
    return_index : bool, optional
        If True, also return the indices of the original array that give the unique rows.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used to reconstruct
        the original array.
    return_counts : bool, optional
        If True, also return the number of times each unique row appears in the original array.
    Returns
    -------
    unique : ndarray
        The sorted unique rows of the array.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique rows in the original array.
        Only provided if return_index is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the unique array.
        Only provided if return_inverse is True.
    unique_counts : ndarray, optional
        The number of times each unique row appears in the original array.
        Only provided if return_counts is True.
    Raises
    ------
    ValueError
        If the input array is not 2D.
    Notes
    -----
    This is a Numba-compatible version adapted from a CuPy implementation, designed
    to efficiently find unique rows in a 2D array.

    adapted version of https://stackoverflow.com/questions/58662085/is-there-a-cupy-version-supporting-axis-option-in-cupy-unique-function-any
    enhanced with functionality from original cupy unique functions https://github.com/cupy/cupy/blob/v7.5.0/cupy/manipulation/add_remove.py#L19
    changed to numba compatible version: https://github.com/numba/numba/issues/7663
    """


    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")


    # don't want to sort original data
    sortarr = array.copy()

    # so we can remember the original indexes of each row
    perm = np.array([i for i in range(array.shape[0])])


    for i in range(sortarr.shape[1] - 1, -1, -1):
        sorter = sortarr[:, i].argsort(kind="mergesort")  # mergesort to keep associations
        sortarr = sortarr[sorter]
        perm = perm[sorter]

    mask        = np.empty(array.shape[0], dtype=np.bool_)
    mask[0]     = True
    mask[1:]    = nb_any_axis1(sortarr[1:] != sortarr[:-1])

    retu = sortarr[mask]


    if return_index:
        retidx = perm[mask]
    else:
        retidx = None

    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        retinv = inv_idx
    else:
        retinv = None

    if return_counts:
        nonzero = np.nonzero(mask)[0]  # may synchronize
        idx = np.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        retc = idx[1:] - idx[:-1]
    else:
        retc = None

    # return ret, retidx, retinv, retc
    return retu, retidx, retinv, retc


@jit(nopython=True)
def nb_float_to_string(float_to_string):
    """
    Convert a float to a string using numba's objmode.

    This function is designed to be used within numba JIT-compiled code to perform the conversion
    of a float value to a string, which is not directly supported in numba's nopython mode.

    Parameters
    ----------
    float_to_string : float
        The float value to be converted to a string.

    Returns
    -------
    str
        The string representation of the input float.

    Notes
    -----
    Uses numba's objmode to temporarily exit nopython mode, allowing the use of Python's
    string formatting which is not available in nopython mode.
    """
    with objmode(string=types.unicode_type):  # declare that the "escaping" string variable is of unicode type.
        string = f'{float_to_string}'
    return string