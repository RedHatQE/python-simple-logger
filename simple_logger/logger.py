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

    def _log_acl(self) -> tuple[int, int | None, int | None]:
        mode = 0o600
        base_uid: int | None = None
        base_gid: int | None = None
        with suppress(OSError):
            base_stat = os.stat(self.baseFilename)
            base_mode = stat.S_IMODE(base_stat.st_mode)
            if base_mode & stat.S_IWGRP:
                mode |= 0o060
            if base_mode & stat.S_IWOTH:
                mode |= 0o006
            base_uid = base_stat.st_uid
            base_gid = base_stat.st_gid
        return mode, base_uid, base_gid

    def _lock_mode(self) -> int:
        return self._log_acl()[0]

    def _open_fallback_lock_fd(
        self,
        lock_mode: int,
        base_uid: int | None,
        base_gid: int | None,
        *,
        create: bool = True,
    ) -> int | None:
        uid = os.getuid() if hasattr(os, "getuid") else None
        shared_gid = base_gid if (lock_mode & 0o060) else None
        if shared_gid is not None:
            dir_name = f"simple-logger-gid-{shared_gid}"
            dir_mode = 0o770
            perm_mask = 0o007
            file_mode = lock_mode & 0o660
        else:
            dir_name = f"simple-logger-{uid}" if uid is not None else "simple-logger"
            dir_mode = 0o700
            perm_mask = 0o077
            file_mode = 0o600

        digest = hashlib.sha256(self.baseFilename.encode()).hexdigest()[:16]
        fallback_dir = os.path.join(tempfile.gettempdir(), dir_name)
        fallback_path = os.path.join(fallback_dir, f"{digest}.lock")
        if not create and not os.path.exists(fallback_path):
            return None

        try:
            if create:
                os.makedirs(fallback_dir, mode=dir_mode, exist_ok=True)
                if shared_gid is not None and hasattr(os, "chown"):
                    with suppress(OSError):
                        os.chown(fallback_dir, -1, shared_gid)
                        os.chmod(fallback_dir, dir_mode)
            d_st = os.lstat(fallback_dir)
        except OSError:
            return None

        if not stat.S_ISDIR(d_st.st_mode) or (d_st.st_mode & perm_mask) != 0:
            return None
        allowed_uids = {u for u in (uid, base_uid) if u is not None}
        if shared_gid is not None:
            if d_st.st_gid != shared_gid or (uid is not None and d_st.st_uid not in allowed_uids):
                return None
        elif uid is not None and d_st.st_uid != uid:
            return None

        no_follow = getattr(os, "O_NOFOLLOW", 0)
        flags = (os.O_CREAT | os.O_RDWR if create else os.O_RDWR) | no_follow
        try:
            try:
                fd = os.open(fallback_path, flags, file_mode)
            except PermissionError:
                if not create or os.path.exists(fallback_path):
                    fd = os.open(fallback_path, os.O_RDONLY | no_follow)
                else:
                    raise
        except OSError:
            return None

        if shared_gid is not None:
            if hasattr(os, "fchown"):
                with suppress(OSError):
                    os.fchown(fd, -1, shared_gid)
            if hasattr(os, "fchmod"):
                with suppress(OSError):
                    os.fchmod(fd, file_mode)

        try:
            f_st = os.fstat(fd)
            if not stat.S_ISREG(f_st.st_mode):
                with suppress(OSError):
                    os.close(fd)
                return None
            if shared_gid is not None:
                if f_st.st_gid != shared_gid or (uid is not None and f_st.st_uid not in allowed_uids):
                    with suppress(OSError):
                        os.close(fd)
                    return None
            elif uid is not None and f_st.st_uid != uid:
                with suppress(OSError):
                    os.close(fd)
                return None
        except OSError:
            with suppress(OSError):
                os.close(fd)
            return None

        self._active_lock_path = fallback_path
        return fd

    def _get_lock_fd(self) -> int | None:
        pid = os.getpid()
        if self._lock_fd is not None and self._lock_pid != pid:
            self._close_lock_fd()
        if self._lock_fd is None:
            lock_mode, base_uid, base_gid = self._log_acl()
            fd: int | None = None
            if not os.path.exists(self._lock_path):
                fd = self._open_fallback_lock_fd(lock_mode, base_uid, base_gid, create=False)
            if fd is None:
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
                            if (lock_mode & 0o060) and base_gid is not None and hasattr(os, "fchown"):
                                with suppress(OSError):
                                    os.fchown(fd, -1, base_gid)
                            if hasattr(os, "fchmod"):
                                with suppress(OSError):
                                    os.fchmod(fd, lock_mode)
                        self._active_lock_path = self._lock_path
                        break
                except OSError:
                    if os.path.exists(self._lock_path):
                        if self.maxBytes > 0:
                            raise
                        return None
                    fd = None

            if fd is None:
                if os.path.exists(self._lock_path):
                    return None
                fd = self._open_fallback_lock_fd(lock_mode, base_uid, base_gid, create=True)
                if fd is None:
                    return None

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
        if fcntl is None or self.maxBytes <= 0:
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
