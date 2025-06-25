# logger.py

import logging
import os

def get_logger(name=__name__, log_file='training_logs/train.log', level=logging.INFO):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file = logging.FileHandler(log_file)
    file.setFormatter(formatter)
    logger.addHandler(file)

    return logger
