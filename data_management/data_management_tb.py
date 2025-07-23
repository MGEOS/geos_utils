"""
Data management helper functions.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------

"""


import errno
import os


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