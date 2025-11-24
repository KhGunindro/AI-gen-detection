import logging
import os

def setup_logger(cfg, run_dir=None):

    log_cfg = cfg.get("logging", {})

    level = log_cfg.get("level", "INFO").upper()
    log_to_file = log_cfg.get("log_to_file", True)
    log_format = log_cfg.get("format", "[%(asctime)s] [%(levelname)s] %(message)s")
    filename = log_cfg.get("filename", "project.log")

    level = getattr(logging, level, logging.INFO)

    logger = logging.getLogger("deepfake_classifier")
    logger.setLevel(level)
    logger.handlers = [] 

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_path = filename

        if run_dir:
            log_path = os.path.join(run_dir, filename)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
