import logging


def setup_logger(log_file: str):
    # Create a logger object
    logger = logging.getLogger("yamyam")
    logger.setLevel(logging.DEBUG)

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file, mode="w")

    # Create a console handler to log messages to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # You can set different level if needed

    # Set a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
