from __future__ import annotations
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

from colorlog import ColoredFormatter

LOGGER = logging.getLogger(__name__)
LOGGERS: Dict[str, logging.Logger] = {}
SUCCESS: int = 32
HASH: int = 33


class DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        self.repeated_number: int

        repeated_number_exists: bool | None = getattr(self, "repeated_number", None)
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            if repeated_number_exists:
                LOGGER.warning(f"Last log repeated {self.repeated_number} times.")

            self.repeated_number = 0
            return True
        if repeated_number_exists:
            self.repeated_number += 1
        else:
            self.repeated_number = 1
        return False


class WrapperLogFormatter(ColoredFormatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        return datetime.fromtimestamp(record.created).isoformat()


class SimpleLogger(logging.getLoggerClass()):  # type: ignore[misc]
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        super().__init__(name=name, level=level)

        logging.addLevelName(SUCCESS, "SUCCESS")
        logging.addLevelName(HASH, "HASH")

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(SUCCESS, msg, *args, **kwargs)

    def hash(self, msg: str, *args: Any, **kwargs: Any) -> None:
        for arg in args:
            msg = msg.replace(arg, "*****")
        args = ()
        self.log(HASH, msg, *args, **kwargs)


logging.setLoggerClass(SimpleLogger)


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    filename: str | None = None,
    console: bool = True,
    file_max_bytes: int = 104857600,
    file_backup_count: int = 20,
) -> logging.Logger:
    """
    Get logger object for logging.

    Args:
        name (str):): name of the logger
        level (int or str): level of logging
        filename (str): filename (full path) for log file
        console (bool): whether to log to console
        file_max_bytes (int): log file size max size in bytes
        file_backup_count (int): max number of log files to keep

    Returns:
        Logger: logger object

    """
    if LOGGERS.get(name):
        return LOGGERS[name]

    logger_obj = logging.getLogger(name)
    log_formatter = WrapperLogFormatter(
        fmt="%(asctime)s %(name)s %(log_color)s%(levelname)s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "SUCCESS": "bold_green",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
            "HASH": "bold_yellow",
        },
        secondary_log_colors={},
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt=log_formatter)
        console_handler.addFilter(filter=DuplicateFilter())

        logger_obj.addHandler(hdlr=console_handler)

    logger_obj.setLevel(level=level)
    logger_obj.addFilter(filter=DuplicateFilter())

    if filename:
        log_handler = RotatingFileHandler(filename=filename, maxBytes=file_max_bytes, backupCount=file_backup_count)
        log_handler.setFormatter(fmt=log_formatter)
        log_handler.setLevel(level=level)
        logger_obj.addHandler(hdlr=log_handler)

    logger_obj.propagate = False
    LOGGERS[name] = logger_obj
    return logger_obj
