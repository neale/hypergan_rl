import logging


def get_logger(title, fn):
    logger = logging.getLogger(title + '.log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
                                  datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler('{}.log'.format(fn))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger
