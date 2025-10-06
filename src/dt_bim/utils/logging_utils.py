import logging, sys

def get_logger(name: str = "dt-bim"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
