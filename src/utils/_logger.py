import logging

def get_module_logger(
        # Logger
        name="main",
        # File handler
        filename=".logs/log.log",
        mode='a',
        file_level = 10,
        # Console handler
        console_level = 20):
    # Initialize logger
    logger = logging.getLogger(name)

    # Setup formatters
    formatter_longform = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter_shortform = logging.Formatter('%(levelname)s - %(message)s')

    # Setup stdout handler
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formatter_shortform)
    handler_console.setLevel(console_level)

    # Setup logfile handler
    handler_logfile = logging.FileHandler(filename=filename, mode=mode)
    handler_logfile.setFormatter(formatter_longform)
    handler_logfile.setLevel(file_level)

    # Add handlers
    logger.addHandler(handler_console)
    logger.addHandler(handler_logfile)

    return logger