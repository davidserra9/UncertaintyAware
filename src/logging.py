import os
import time
import logging
import sys

class CustomFormatter(logging.Formatter):

    # grey = "\x1b[38;20m"
    # yellow = "\x1b[33;20m"
    # red = "\x1b[31;20m"
    # bold_red = "\x1b[31;1m"
    # reset = "\x1b[0m"
    format = "%(asctime)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: format,
        logging.WARNING: format,
        logging.ERROR:format,
        logging.CRITICAL: format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create the root folder if it does not exist
    root_folder = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(root_folder):
        print("Creating root folder for logs")
        os.makedirs(root_folder)

    # Create subfolder for the current run
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_folder = os.path.join(root_folder, timestamp)
    os.makedirs(log_folder)
    logger.output_path = log_folder

    # Set up the log file path
    log_file = os.path.join(log_folder, "log.log")

    # Configure the file handler with the log file path
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(CustomFormatter())

    # Configure the console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())

    # Add both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = set_logging()