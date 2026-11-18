import logging
import multiprocessing
import os
import tempfile
import time
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

from simple_logger import logger as logger_module
from simple_logger.logger import MultiProcessRotatingFileHandler, get_logger


@pytest.fixture
def tmp_log_file(tmp_path_factory):
    return tmp_path_factory.mktemp("data") / "test.log"


@pytest.fixture
def basic_logger(tmp_log_file):
    return get_logger(name="test_basic_logger", filename=tmp_log_file)


@pytest.fixture
def logger_with_mask(tmp_log_file):
    return get_logger(name="test_logger_with_mask", filename=tmp_log_file, mask_sensitive=True)


@pytest.fixture
def logger_with_mask_and_pattarns(tmp_log_file):
    return get_logger(
        name="test_logger_with_mask_and_pattarns",
        filename=tmp_log_file,
        mask_sensitive=True,
        mask_sensitive_patterns=["pt1", "pt2"],
    )


def test_mask_sensitive(tmp_log_file, logger_with_mask, logger_with_mask_and_pattarns):
    logger_with_mask.info("My Password Pass!Word123 for all")
    logger_with_mask.info("My Token\n Tok8n!123 for all")

    logger_with_mask_and_pattarns.info("My pt1 Pass!Word123 for all")
    logger_with_mask_and_pattarns.info("My Pt2\n Tok8n!123 for all")

    with open(tmp_log_file) as fd:
        content = fd.read()

    assert "My password *****  for all" in content
    assert "My token *****  for all" in content

    assert "My pt1 *****  for all" in content
    assert "My pt2 *****  for all" in content


def test_existing_logger():
    logger1 = get_logger(name="test_logger")
    logger2 = get_logger(name="test_logger")
    assert logger1 == logger2


def test_disable_console_logging():
    logger = get_logger(name="test_logger", console=False)
    assert isinstance(logger, logging.Logger)


def test_levels(tmp_log_file, basic_logger):
    basic_logger.info("Info message")
    basic_logger.debug("Debug message")
    basic_logger.warning("Warning message")
    basic_logger.error("Error message")
    basic_logger.critical("Critical message")
    basic_logger.success("Success message")
    basic_logger.success("Success message")
    basic_logger.success("Success message")
    basic_logger.success("Success message")
    basic_logger.success("Success message")
    basic_logger.info("last message")

    with open(tmp_log_file) as fd:
        content = fd.read()

    assert "Info message" in content
    assert "Debug message" not in content
    assert "Warning message" in content
    assert "Error message" in content
    assert "Critical message" in content
    assert "Success message" in content
    assert "repeated 4 times" in content


def test_duplicate_filter_with_percent_placeholders(tmp_log_file):
    logger = get_logger(name="test_duplicate_filter_percent", filename=tmp_log_file, level=logging.DEBUG)
    url = "http://example.com"
    job_id = "job-123"

    for _ in range(5):
        logger.debug("Base URL from Host header: %s", url)

    logger.info("Report generated for job_id: %s", job_id)

    with open(tmp_log_file) as fd:
        content = fd.read()

    assert "repeated 4 times" in content
    assert f"Report generated for job_id: {job_id}" in content
    assert "Last log `Base URL from Host header: %s`" in content


def _make_record(msg: str = "hello world", level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def _fork_worker(handler: MultiProcessRotatingFileHandler, worker_id: int, count: int) -> None:
    for idx in range(count):
        handler.emit(_make_record(f"worker={worker_id} idx={idx} {'x' * 40}"))


def _cleanup_processes(procs: list[multiprocessing.process.BaseProcess]) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)


def test_get_logger_uses_multiprocess_handler(tmp_log_file):
    log = get_logger(name="test_mp_handler_type", filename=tmp_log_file, console=False)
    file_handlers = [h for h in log.handlers if isinstance(h, MultiProcessRotatingFileHandler)]
    assert len(file_handlers) == 1
    file_handlers[0].close()


def test_multiprocess_fork_and_rollover(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=250, backupCount=3)
    try:
        # Open the lock FD in parent before forking to verify child PID reopens its own lock FD.
        handler.emit(_make_record("parent-init"))
        parent_lock_fd = handler._lock_fd
        assert parent_lock_fd is not None
        assert handler._lock_pid == os.getpid()

        ctx = multiprocessing.get_context("fork")
        procs = [ctx.Process(target=_fork_worker, args=(handler, w, 15)) for w in range(4)]
        try:
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=15)
                assert proc.exitcode == 0
        finally:
            _cleanup_processes(procs)
    finally:
        handler.close()

    assert handler._lock_fd is None
    assert handler._lock_pid is None

    # Active log + backups should contain valid intact lines and at least one rollover occurred.
    assert (tmp_log_file.parent / "test.log.1").exists()
    all_lines: list[str] = []
    for suffix in ["", ".1", ".2", ".3"]:
        path = tmp_log_file.parent / f"test.log{suffix}"
        if path.exists():
            all_lines.extend(path.read_text().splitlines())
    assert "parent-init" in all_lines or any("worker=" in line for line in all_lines)
    for line in all_lines:
        assert line == "parent-init" or line.startswith("worker=")


def test_reopen_if_rotated_and_missing_file(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("before-external-rotate"))
        rotated_path = tmp_log_file.with_name("test.log.external")
        os.rename(tmp_log_file, rotated_path)

        handler.emit(_make_record("after-external-rotate"))
        assert "after-external-rotate" in tmp_log_file.read_text()
        assert "after-external-rotate" not in rotated_path.read_text()

        # Simulate deleted log file (FileNotFoundError in _reopen_if_rotated)
        os.unlink(tmp_log_file)
        handler._reopen_if_rotated()
        assert tmp_log_file.exists()
    finally:
        handler.close()


def test_reopen_if_rotated_noop_when_closed_or_invalid_stream(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, delay=True)
    try:
        handler._reopen_if_rotated()
        assert handler.stream is None

        handler.stream = MagicMock()
        handler.stream.fileno.side_effect = ValueError("closed")
        handler._reopen_if_rotated()
    finally:
        handler.stream = None
        handler.close()


def test_should_rollover_edge_cases(tmp_log_file, monkeypatch):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=50, backupCount=2, delay=True)
    try:
        # Empty file with delay=True: opens stream and never rolls over an empty file.
        assert handler.shouldRollover(_make_record("hello")) is False
        assert handler.stream is not None

        # Non-regular file check returns False when file is non-empty.
        tmp_log_file.write_text("1234567890")
        monkeypatch.setattr(os.path, "isfile", lambda _p: False)
        assert handler.shouldRollover(_make_record("x" * 50)) is False
        monkeypatch.undo()

        # External write pushes disk size over maxBytes -> True even though this handler wrote nothing.
        tmp_log_file.write_text("x" * 45)
        assert handler.shouldRollover(_make_record("123456")) is True

        # maxBytes == 0 disables rollover.
        handler.maxBytes = 0
        assert handler.shouldRollover(_make_record("x" * 100)) is False
    finally:
        handler.close()


def test_do_rollover_skips_below_max_bytes(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=100, backupCount=2)
    try:
        handler.emit(_make_record("small-line"))
        assert 0 < tmp_log_file.stat().st_size < 100
        handler.doRollover()
        assert not tmp_log_file.with_name("test.log.1").exists()
        assert tmp_log_file.read_text() == "small-line\n"
    finally:
        handler.close()


def test_do_rollover_skips_when_already_rotated_empty_or_missing(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=50, backupCount=2)
    try:
        # Empty file: doRollover skips rotation.
        handler.doRollover()
        assert not tmp_log_file.with_name("test.log.1").exists()

        # Missing file: doRollover reopens and skips rotation.
        os.unlink(tmp_log_file)
        handler.doRollover()
        assert tmp_log_file.exists()
        assert not tmp_log_file.with_name("test.log.1").exists()

        # Another process already rotated before doRollover acquired lock (inode changed).
        handler.emit(_make_record("initial"))
        os.rename(tmp_log_file, tmp_log_file.with_name("test.log.1"))
        external_content = f"from-other-process-{'x' * 50}\n"
        tmp_log_file.write_text(external_content)
        handler.doRollover()
        assert tmp_log_file.read_text() == external_content
        assert tmp_log_file.with_name("test.log.1").read_text() == "initial\n"
    finally:
        handler.close()


def test_do_rollover_suppresses_file_not_found_error(tmp_log_file, monkeypatch):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=3)
    try:
        handler.emit(_make_record("line-one-long-enough"))
        assert tmp_log_file.stat().st_size >= 20

        monkeypatch.setattr(handler, "rotate", MagicMock(side_effect=FileNotFoundError("vanished")))
        handler.doRollover()
        assert handler.stream is not None
    finally:
        handler.close()


def test_do_rollover_propagates_oserror_on_backup_shift(tmp_log_file, monkeypatch):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=2)
    try:
        handler.emit(_make_record("line-one-long-enough"))
        handler.emit(_make_record("line-two-long-enough"))
        backup_one = tmp_log_file.with_name("test.log.1")
        assert backup_one.read_text() == "line-one-long-enough\n"

        monkeypatch.setattr(os, "replace", MagicMock(side_effect=PermissionError("disk error")))
        with pytest.raises(PermissionError, match="disk error"):
            handler.doRollover()
        assert backup_one.read_text() == "line-one-long-enough\n"

        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler, "handleError", mock_handle_error)
        record = _make_record("triggers-backup-shift-error")
        handler.emit(record)
        mock_handle_error.assert_called_once_with(record)
        assert backup_one.read_text() == "line-one-long-enough\n"
    finally:
        handler.close()


def test_do_rollover_propagates_oserror_on_rotate(tmp_log_file, monkeypatch):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=1)
    try:
        handler.emit(_make_record("line-one-long-enough"))
        handler.emit(_make_record("line-two-long-enough"))
        backup_one = tmp_log_file.with_name("test.log.1")
        assert backup_one.read_text() == "line-one-long-enough\n"

        monkeypatch.setattr(handler, "rotate", MagicMock(side_effect=OSError("disk error")))
        with pytest.raises(OSError, match="disk error"):
            handler.doRollover()
        assert backup_one.read_text() == "line-one-long-enough\n"

        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler, "handleError", mock_handle_error)
        record = _make_record("triggers-rotate-error")
        handler.emit(record)
        mock_handle_error.assert_called_once_with(record)
        assert backup_one.read_text() == "line-one-long-enough\n"
    finally:
        handler.close()


def test_lock_fallback_and_no_lock_preserves_writes(tmp_log_file, monkeypatch):
    real_open = os.open

    def fail_primary_lock(path, flags, mode=0o777, *args, **kwargs):
        if str(path) == f"{tmp_log_file}.lock":
            raise PermissionError("lock dir read-only")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_primary_lock)
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    fallback_path = None
    try:
        handler.emit(_make_record("written-with-fallback-lock"))
        fallback_path = handler._active_lock_path
        assert fallback_path != handler._lock_path
        assert fallback_path.startswith(tempfile.gettempdir())
        assert handler._lock_fd is not None
        assert "written-with-fallback-lock" in tmp_log_file.read_text()
    finally:
        handler.close()
        if fallback_path and os.path.exists(fallback_path):
            os.unlink(fallback_path)

    def fail_all_locks(path, flags, mode=0o777, *args, **kwargs):
        if str(path).endswith(".lock"):
            raise OSError("no lock available")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_all_locks)
    handler_no_lock = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        assert handler_no_lock._get_lock_fd() is None
        handler_no_lock.emit(_make_record("written-without-lock"))
        content = tmp_log_file.read_text()
        assert "written-with-fallback-lock" in content
        assert "written-without-lock" in content
    finally:
        handler_no_lock.close()


def test_process_lock_reopens_when_unlinked_or_replaced(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("first"))
        lock_path = tmp_log_file.with_name("test.log.lock")

        # Unlinking lock file triggers FileNotFoundError re-validation and recreates the lock.
        os.unlink(lock_path)
        handler.emit(_make_record("after-unlink"))
        assert lock_path.exists()

        # Replacing lock file while handler holds the old FD guarantees a new inode and re-validation.
        old_ino = os.fstat(handler._lock_fd).st_ino
        os.unlink(lock_path)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.close(fd)
        replaced_ino = lock_path.stat().st_ino
        assert replaced_ino != old_ino

        handler.emit(_make_record("after-replace"))
        assert handler._lock_fd is not None
        assert os.fstat(handler._lock_fd).st_ino == replaced_ino

        content = tmp_log_file.read_text()
        assert "first" in content
        assert "after-unlink" in content
        assert "after-replace" in content
    finally:
        handler.close()


def test_multiprocess_lock_unlink_recreate_during_rollover(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=300, backupCount=30)
    lock_path = f"{tmp_log_file}.lock"
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_fork_worker, args=(handler, w, 20)) for w in range(4)]
    try:
        try:
            for proc in procs:
                proc.start()
            deadline = time.monotonic() + 10
            while any(proc.is_alive() for proc in procs) and time.monotonic() < deadline:
                with suppress(FileNotFoundError):
                    os.unlink(lock_path)
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
                os.close(fd)
                time.sleep(0.002)
            for proc in procs:
                proc.join(timeout=15)
                assert proc.exitcode == 0
        finally:
            _cleanup_processes(procs)
    finally:
        handler.close()

    assert (tmp_log_file.parent / "test.log.1").exists()
    all_lines: list[str] = []
    for path in tmp_log_file.parent.glob("test.log*"):
        if path.name != "test.log.lock":
            all_lines.extend(path.read_text().splitlines())

    expected_lines = {f"worker={w} idx={i} {'x' * 40}" for w in range(4) for i in range(20)}
    assert len(all_lines) == len(expected_lines)
    assert set(all_lines) == expected_lines


def test_non_posix_fallback_and_emit_error_handling(tmp_log_file, monkeypatch):
    monkeypatch.setattr(logger_module, "fcntl", None)
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=30, backupCount=2)
    try:
        handler.emit(_make_record("first-record-exceeds-limit"))
        handler.emit(_make_record("second-record"))
        assert tmp_log_file.with_name("test.log.1").exists()

        monkeypatch.setattr(handler, "shouldRollover", MagicMock(side_effect=RuntimeError("boom")))
        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler, "handleError", mock_handle_error)
        record = _make_record("triggers-error")
        handler.emit(record)
        mock_handle_error.assert_called_once_with(record)
    finally:
        handler.close()


def test_step_and_hash_levels(tmp_log_file):
    log = get_logger(name="test_step_hash_logger", filename=tmp_log_file, console=False)
    try:
        log.step("Step message")
        log.hash("Secret value mysecret123 here", hash=["mysecret123"])
        content = tmp_log_file.read_text()
        assert "STEP" in content
        assert "Step message" in content
        assert "HASH" in content
        assert "mysecret123" not in content
        assert "*****" in content
    finally:
        for handler in list(log.handlers):
            handler.close()
        logger_module.LOGGERS.pop("test_step_hash_logger", None)
