from . import NDateTime

DATETIME = NDateTime()

LOG_LEVELS = {"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5}


class Logger:
    """Logger class to log messages to the console.
    Levels:
    - DEBUG (default): Detailed information, typically of interest only when diagnosing problems.
    - INFO: Confirmation that things are working as expected.
    - WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
    - ERROR: Due to a more serious problem, the software has not been able to perform some function.
    - CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    """

    def __init__(self):
        self.level = LOG_LEVELS.get("DEBUG")

    def log(self, message):
        """Log a message to the console."""
        logItem = (
            "{"
            + "level: 'DEBUG', "
            + "message: '"
            + message
            + "', "
            + "timestamp: '"
            + str(DATETIME.get_current_timestamp())
            + "'"
            + "}"
        )
        print(logItem)
