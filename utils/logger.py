"""
# Author = ruben
# Date: 23/4/24
# Project: RetiNNAR
# File: logger.py

Description: Defines project log
"""

import logging


class Logger:

    def __init__(self):
        self.logger = None
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s',
                                      datefmt='%Y-%m-%d_%H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log(self):
        return self.logger


if __name__ == '__main__':
    logger = Logger().log()
    logger.error("Logger test")
