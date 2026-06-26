import logging

def setup_logger(level=logging.DEBUG, name="chat_model", colorize: bool = True, formatter = None) -> logging.Logger:
    """Sets up a logger for the application.
    
    Args:
        level (int): Logging level. Default is logging.DEBUG.
        name (str): Logger name. Default is "chat_model".
        colorize (bool): Whether to colorize log messages. Default is True.
        formatter (logging.Formatter): Custom formatter. Default is None.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') if formatter is None else formatter

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger
