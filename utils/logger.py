import logging
import os

def _set_logger(log_name):
    formater = logging.Formatter('%(levelname)-8s: %(message)s')
    logger = logging.getLogger('log_name')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(levelname) -8s: %(message)s'))
    console_handler.setLevel(logging.INFO)

    log_path = os.path.join("../data/log", log_name+".log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s'))
    file_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger