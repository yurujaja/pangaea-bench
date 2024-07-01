import logging
import os


def setup_logger(log_file, level=logging.DEBUG):

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logging
    if logger.handlers:
        logger.handlers = []

    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter.datefmt = "%m/%d/%Y %H:%M:%S"
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
