"""
Data management helper functions.
--------------------------
author: Matthias Gassilloud
date: 04.08.2025
--------------------------

"""


import errno
import os
import glob
import numpy as np
from pathlib import Path


### file/folder management
def check_file_exists(file_path):

    if not os.path.isfile(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

def check_dir_exists(dir_path):

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dir_path)

def mkdir_if_missing(dir_path):

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def delete_file(file_path):

    try:
        os.remove(file_path)
    except OSError:
        pass

def find_filenames(base_dir, regex_pattern="*.las"):
    """
    Find filenames in the specified directory matching the given regex pattern.
    
    
    Parameters
    ----------
    base_dir : str
        The base directory to search in.
    pattern : str, optional
        The glob pattern to match filenames, by default "*.las".
    
    Returns
    -------
    list
        List of full paths to matching files.
    """
    return glob.glob(os.path.join(base_dir, regex_pattern))

def print_array_size_gb(array: np.ndarray):
    
    size_in_bytes = array.nbytes
    size_in_gigabytes = size_in_bytes / (1024**3)
    print(f"Size of the array: {size_in_gigabytes:.2f} GB")

def get_file_size_gb(file_path):
    """get file size in GB

    Parameters
    ----------
    file_path : str
        path to file

    Returns
    -------
    float
        file size in GB
    """

    file_path = Path(file_path)
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024**3)


### file type conversion
def df_instances_to_dict(df):
    """
    Convert pandas dataframe to dictionary, excluding NaN values.
    
    This function converts a pandas DataFrame to a list of dictionaries where each dictionary
    represents a row. Cells with NaN values or None are excluded from the resulting dictionaries.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert to a dictionary.
    
    Returns
    -------
    list
        List of dictionaries, where each dictionary represents a row from the DataFrame
        with NaN values excluded.
    """
    return [{k: v for k, v in m.items() if v == v and v is not None} for m in df.to_dict(orient='records')]


### numpy
def get_suitable_np_datatype(value, dtype="uint"):
    '''
    implemented with value < np.iinfo(dt).max to be aible to set nan = np.iinfo(dt).max
    '''

    if dtype == "uint":
        dtype_list = [np.uint8, np.uint16, np.uint32, np.uint64]
    elif dtype == "int":
        dtype_list = [np.int8, np.int16, np.int32, np.int64]
    elif dtype == "float":
        dtype_list = [np.float16, np.float32, np.float64]

    if dtype == "uint" or dtype == "int":
        for dt in dtype_list:
            if np.iinfo(dt).min <= value < np.iinfo(dt).max:
                return dt
        print(f"Cannot represent {value} with {np.iinfo(dt)}")

    elif dtype == "float":
        for dt in dtype_list:
            if np.finfo(dt).min <= value < np.finfo(dt).max:
                return dt
        print(f"Cannot represent {value} with {np.finfo(dt)}")

    return None


### memory mapping
def read_memmap_array(array_mapped_path, array_shape, array_dtype):
    '''
    store array, return mapped array. original array can be closed afterwards.
    '''

    array_mapped = np.memmap(array_mapped_path,
                        shape=array_shape,
                        mode = 'r',
                        dtype = array_dtype)
    return array_mapped

def write_memmap_array(array, array_mapped_path):
    '''
    store array, return mapped array. original array can be closed afterwards.
    '''
    array_shape = array.shape
    array_dtype = array.dtype
    array.tofile(array_mapped_path)  # should be .rbf
    array_mapped = np.memmap(array_mapped_path,
                        shape=array_shape,
                        mode = 'r',
                        dtype = array_dtype)
    return array_mapped