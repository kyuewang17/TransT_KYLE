from __future__ import absolute_import, print_function

import os
import time
import sys
import logging
import numpy as np
import torch
import random
from loguru import logger as logger_u
from typing import Dict


class Timer():
    r"""
    Mesure & print elapsed time within environment
    """
    def __init__(self,
                 name: str = "",
                 output_dict: Dict = None,
                 verbose: bool = False):
        """Timing usage
        Parameters
        ----------
        name : str, optional
            name of timer, used in verbose & output_dict, by default ''
        output_dict : Dict, optional
            dict-like object to receive elapsed time in output_dict[name], by default None
        verbose : bool, optional
            verbose or not via logger, by default False
        """
        self.name = name
        self.output_dict = output_dict
        self.verbose = verbose

    def __enter__(self, ):
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.time()
        elapsed_time = self.toc - self.tic
        if self.output_dict is not None:
            self.output_dict[self.name] = elapsed_time
        if self.verbose:
            print_str = '%s elapsed time: %f' % (self.name, elapsed_time)
            logger_u.info(print_str)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def fix_seed(random_seed, logger=None):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if logger is not None:
        logger.info("Random Seed Fixed as [{}]...!".format(random_seed))


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
