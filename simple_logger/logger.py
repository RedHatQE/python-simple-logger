from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from colorlog import ColoredFormatter

fcntl: Any
try:
    import fcntl
except ImportError:
    fcntl = None

LOGGERS: dict[str, logging.Logger] = {}
SUCCESS: int = 32
HASH: int = 33
STEP: int = 34


class DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        self.repeated_number: int
        self.last_log: tuple[str, int, str]

        _repeated_number: int = getattr(self, "repeated_number", 0)
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            if _repeated_number > 1:
                # Escape % to prevent %-style formatting from treating them as placeholders
                _last_log_msg = self.last_log[-1].replace("%", "%%")
                record.msg = f"{record.msg} --- [DuplicateFilter: Last log `{_last_log_msg}` repeated {self.repeated_number} times]"

            self.last_log = current_log
            self.repeated_number = 0
            return True
        else:
            self.repeated_number += 1
            return False

    def redact(self, msg: str) -> str:
        return msg


class RedactingFilter(logging.Filter):
    def __init__(self, patterns: list[str]):
        super().__init__()
        self._patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.msg)
        return True

    def redact(self, msg: str) -> str:
        for pattern in self._patterns:
            msg = re.sub(rf"({pattern}\W+[^\s+]+)", f"{pattern} {'*' * 5} ", msg, flags=re.IGNORECASE)
        return msg


class WrapperLogFormatter(ColoredFormatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()


class MultiProcessRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler safe for concurrent writes and rotations across processes."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        self._lock_fd: int | None = None
        self._lock_pid: int | None = None
        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )

    def _get_lock_fd(self) -> int:
        pid = os.getpid()
        if self._lock_fd is not None and self._lock_pid != pid:
            self._close_lock_fd()
        if self._lock_fd is None:
            self._lock_fd = os.open(f"{self.baseFilename}.lock", os.O_CREAT | os.O_RDWR, 0o666)
            self._lock_pid = pid
        return self._lock_fd

    def _close_lock_fd(self) -> None:
        if self._lock_fd is not None:
            with suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None
            self._lock_pid = None

    @contextmanager
    def _process_lock(self) -> Generator[None, None, None]:
        if fcntl is None:
            yield
            return
        fd = self._get_lock_fd()
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)

    def _reopen_if_rotated(self) -> None:
        if self.stream is None:
            return
        try:
            stream_stat = os.fstat(self.stream.fileno())
        except (OSError, ValueError):
            return
        try:
            file_stat = os.stat(self.baseFilename)
            rotated = (stream_stat.st_dev, stream_stat.st_ino) != (file_stat.st_dev, file_stat.st_ino)
        except FileNotFoundError:
            rotated = True
        if rotated:
            try:
                self.stream.close()
            finally:
                self.stream = None
            self.stream = self._open()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        self._reopen_if_rotated()
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            size = os.fstat(self.stream.fileno()).st_size
            if not size:
                return False
            if os.path.exists(self.baseFilename) and not os.path.isfile(self.baseFilename):
                return False
            if size + len(f"{self.format(record)}\n") >= self.maxBytes:
                return True
        return False

    def _do_rollover_locked(self) -> None:
        stream_id: tuple[int, int] | None = None
        if self.stream is not None:
            try:
                st = os.fstat(self.stream.fileno())
                stream_id = (st.st_dev, st.st_ino)
            except (OSError, ValueError):
                pass
            try:
                self.stream.close()
            finally:
                self.stream = None

        try:
            file_stat = os.stat(self.baseFilename)
        except FileNotFoundError:
            if not self.delay:
                self.stream = self._open()
            return

        if (stream_id is not None and (file_stat.st_dev, file_stat.st_ino) != stream_id) or file_stat.st_size == 0:
            if not self.delay:
                self.stream = self._open()
            return

        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                if os.path.exists(sfn):
                    with suppress(OSError):
                        if os.path.exists(dfn):
                            with suppress(FileNotFoundError):
                                os.remove(dfn)
                        os.rename(sfn, dfn)
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            with suppress(OSError):
                if os.path.exists(dfn):
                    with suppress(FileNotFoundError):
                        os.remove(dfn)
                self.rotate(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()

    def doRollover(self) -> None:
        self.acquire()
        try:
            with self._process_lock():
                self._do_rollover_locked()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._process_lock():
                if self.shouldRollover(record):
                    self._do_rollover_locked()
                logging.FileHandler.emit(self, record)
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            try:
                super().close()
            finally:
                self._close_lock_fd()
        finally:
            self.release()


class SimpleLogger(logging.getLoggerClass()):  # type: ignore[misc]
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        super().__init__(name=name, level=level)

        logging.addLevelName(SUCCESS, "SUCCESS")
        logging.addLevelName(HASH, "HASH")
        logging.addLevelName(STEP, "STEP")

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(SUCCESS, msg, *args, **kwargs)

    def step(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(STEP, msg, *args, **kwargs)

    def hash(self, msg: str, *args: Any, **kwargs: Any) -> None:
        to_hash: list[str] = kwargs.pop("hash", [])
        for hash in to_hash:
            msg = msg.replace(hash, "*****")

        self.log(HASH, msg, *args, **kwargs)


logging.setLoggerClass(SimpleLogger)


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    filename: str | None = None,
    console: bool = True,
    file_max_bytes: int = 104857600,
    file_backup_count: int = 20,
    mask_sensitive: bool = False,
    duplicate_filter: bool = True,
    mask_sensitive_patterns: list[str] | None = None,
    force_color: bool | None = None,
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
        mask_sensitive (bool): whether to mask sensitive information
        mask_sensitive_patterns (list[str]): list of patterns to mask
        force_color (bool or None): force colored output even in non-TTY environments.
            If None, will check FORCE_COLOR environment variable.

    Returns:
        Logger: logger object

    """
    if LOGGERS.get(name):
        return LOGGERS[name]

    # Determine force_color setting
    if force_color is None:
        # Check FORCE_COLOR environment variable
        force_color_env = os.environ.get("FORCE_COLOR", "").lower()
        force_color = force_color_env in ("1", "true")

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
            "STEP": "bold_cyan",
        },
        secondary_log_colors={},
        force_color=force_color,
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt=log_formatter)
        if duplicate_filter:
            console_handler.addFilter(filter=DuplicateFilter())

        logger_obj.addHandler(hdlr=console_handler)

    logger_obj.setLevel(level=level)
    if duplicate_filter:
        logger_obj.addFilter(filter=DuplicateFilter())

    if mask_sensitive:
        mask_sensitive_patterns = mask_sensitive_patterns or ["password", "token", "apikey", "secret"]
        logger_obj.addFilter(filter=RedactingFilter(patterns=mask_sensitive_patterns))

    if filename:
        log_handler = MultiProcessRotatingFileHandler(
            filename=filename, maxBytes=file_max_bytes, backupCount=file_backup_count
        )
        log_handler.setFormatter(fmt=log_formatter)
        log_handler.setLevel(level=level)
        logger_obj.addHandler(hdlr=log_handler)

    logger_obj.propagate = False
    LOGGERS[name] = logger_obj
    return logger_obj
