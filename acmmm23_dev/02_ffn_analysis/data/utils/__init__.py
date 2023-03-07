from __future__ import absolute_import, print_function

import os
import logging
import numpy as np
import torch


def set_logger(logging_level=logging.INFO, log_name="root", logging_filepath=None):
    # Define Logger
    if log_name == "root":
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name=log_name)

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Formatter
    formatter = logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    # formatter = "[%(levelname)s] | %(asctime)s : %(message)s"

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # stream_handler.setFormatter(CustomFormatter(formatter))
    logger.addHandler(stream_handler)

    # Set File Handler
    if logging_filepath is not None:
        file_handler = logging.FileHandler(logging_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_npy_cvt_torch(npy_filepath, device=None):
    assert npy_filepath.endswith("npy")
    npy_data = np.load(npy_filepath)
    if device is None:
        return torch.from_numpy(npy_data)
    else:
        with torch.cuda.device(device):
            return torch.cuda.FloatTensor(npy_data)
