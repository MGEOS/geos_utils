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

def print_array_size_in_gb(array: np.ndarray):
    
    size_in_bytes = array.nbytes
    size_in_gigabytes = size_in_bytes / (1024 * 1024 * 1024)
    print(f"Size of the array: {size_in_gigabytes:.2f} GB")


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
