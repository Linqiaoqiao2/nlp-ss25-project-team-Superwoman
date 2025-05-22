# logger.py
import logging
import os


def setup_logger(name: str, log_file: str = "rag_pipeline.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)


    if not logger.handlers:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
