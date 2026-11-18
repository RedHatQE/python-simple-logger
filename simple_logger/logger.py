from __future__ import annotations

import codecs
import errno
import hashlib
import logging
import os
import re
import stat
import tempfile
import weakref
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
_UNSUPPORTED_LOCK_ERRNOS: frozenset[int] = frozenset({
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    errno.ENOLCK,
    errno.EINVAL,
})


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
        self.baseFilename = os.path.realpath(self.baseFilename)
        self._lock_path = f"{self.baseFilename}.lock"
        self._active_lock_path = self._lock_path
        _HANDLERS.add(self)

    def _lock_mode(self) -> int:
        mode = 0o600
        with suppress(OSError):
            base_mode = stat.S_IMODE(os.stat(self.baseFilename).st_mode)
            if base_mode & stat.S_IWGRP:
                mode |= 0o060
            if base_mode & stat.S_IWOTH:
                mode |= 0o006
        return mode

    def _get_lock_fd(self) -> int | None:
        pid = os.getpid()
        if self._lock_fd is not None and self._lock_pid != pid:
            self._close_lock_fd()
        if self._lock_fd is None:
            lock_mode = self._lock_mode()
            fd: int | None = None
            try:
                for _ in range(3):
                    try:
                        fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, lock_mode)
                    except FileExistsError:
                        try:
                            try:
                                fd = os.open(self._lock_path, os.O_RDWR)
                            except PermissionError:
                                fd = os.open(self._lock_path, os.O_RDONLY)
                        except FileNotFoundError:
                            continue
                    else:
                        if hasattr(os, "fchmod"):
                            with suppress(OSError):
                                os.fchmod(fd, lock_mode)
                    self._active_lock_path = self._lock_path
                    break
            except OSError:
                if os.path.exists(self._lock_path):
                    return None
                fd = None

            if fd is None:
                if os.path.exists(self._lock_path):
                    return None
                digest = hashlib.sha256(self.baseFilename.encode()).hexdigest()[:16]
                uid = os.getuid() if hasattr(os, "getuid") else None
                dir_name = f"simple-logger-{uid}" if uid is not None else "simple-logger"
                fallback_dir = os.path.join(tempfile.gettempdir(), dir_name)
                try:
                    os.makedirs(fallback_dir, mode=0o700, exist_ok=True)
                    d_st = os.lstat(fallback_dir)
                    if (
                        not stat.S_ISDIR(d_st.st_mode)
                        or (uid is not None and d_st.st_uid != uid)
                        or (d_st.st_mode & 0o077) != 0
                    ):
                        return None
                    fallback_path = os.path.join(fallback_dir, f"{digest}.lock")
                    no_follow = getattr(os, "O_NOFOLLOW", 0)
                    fd = os.open(fallback_path, os.O_CREAT | os.O_RDWR | no_follow, 0o600)
                except OSError:
                    return None
                try:
                    f_st = os.fstat(fd)
                    if not stat.S_ISREG(f_st.st_mode) or (uid is not None and f_st.st_uid != uid):
                        with suppress(OSError):
                            os.close(fd)
                        return None
                except OSError:
                    with suppress(OSError):
                        os.close(fd)
                    return None
                self._active_lock_path = fallback_path

            self._lock_fd = fd
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
        while True:
            fd = self._get_lock_fd()
            if fd is None:
                yield
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                if exc.errno in _UNSUPPORTED_LOCK_ERRNOS:
                    self._close_lock_fd()
                    yield
                    return
                raise
            try:
                fd_stat = os.fstat(fd)
                path_stat = os.stat(self._active_lock_path)
            except FileNotFoundError:
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                self._close_lock_fd()
                continue
            if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                self._close_lock_fd()
                continue
            break
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

    def shouldRollover(self, record: logging.LogRecord, formatted_msg: str | None = None) -> bool:
        self._reopen_if_rotated()
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            size = os.fstat(self.stream.fileno()).st_size
            if not size:
                return False
            if os.path.exists(self.baseFilename) and not os.path.isfile(self.baseFilename):
                return False
            msg = self.format(record) if formatted_msg is None else formatted_msg
            encoding = self.encoding
            if not encoding or encoding == "locale":
                stream_encoding = getattr(self.stream, "encoding", None)
                encoding = stream_encoding if isinstance(stream_encoding, str) else "utf-8"
            line = f"{msg}{self.terminator}"
            if os.linesep != "\n":
                line = line.replace("\n", os.linesep)
            encoder = codecs.getincrementalencoder(encoding)(self.errors or "replace")
            if size > 0:
                with suppress(UnicodeEncodeError):
                    encoder.encode("0")
            msg_bytes = len(encoder.encode(line))
            if size + msg_bytes >= self.maxBytes:
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
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(dfn):
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                    next_dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                    with suppress(FileNotFoundError):
                        os.replace(sfn, next_dfn)
            with suppress(FileNotFoundError):
                if os.name == "nt" and os.path.exists(dfn):
                    os.remove(dfn)
                self.rotate(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()

    def doRollover(self) -> None:
        self.acquire()
        try:
            with self._process_lock():
                try:
                    file_stat = os.stat(self.baseFilename)
                except FileNotFoundError:
                    file_stat = None
                if file_stat is None or self.maxBytes <= 0 or file_stat.st_size < self.maxBytes:
                    self._reopen_if_rotated()
                    if self.stream is None and not self.delay:
                        self.stream = self._open()
                    return
                self._do_rollover_locked()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self.format(record)
            self.acquire()
            try:
                with self._process_lock():
                    if self.shouldRollover(record, formatted):
                        self._do_rollover_locked()
                    if self.stream is None:
                        self.stream = self._open()
                    self.stream.write(formatted + self.terminator)
                    self.flush()
            finally:
                self.release()
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


_HANDLERS: weakref.WeakSet[MultiProcessRotatingFileHandler] = weakref.WeakSet()


def _close_lock_fds_after_fork() -> None:
    for handler in list(_HANDLERS):
        handler._close_lock_fd()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_lock_fds_after_fork)


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
