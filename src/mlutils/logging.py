import logging
from pathlib import Path


def init_logger(logger_name: str, log_path: Path | None = None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_path is not None:
            log_path.touch()
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = init_logger("MLUTILS", None)
