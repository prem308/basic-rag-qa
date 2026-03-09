import logging 
import sys
from functools import lru_cache

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): The logging level to use (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    #create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    #configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    #Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    #console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    #reduce noice from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

@lru_cache
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger instance.
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    Mixin class to provide a logger instance to any class that inherits from it.
    """
    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger instance for the class.

        Returns:
            logging.Logger: A logger instance.
        """
        return get_logger(self.__class__.__name__)

