from __future__ import annotations

import errno
import logging
import multiprocessing
import os
import re
import stat
import tempfile
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from simple_logger import logger as logger_module
from simple_logger.logger import MultiProcessRotatingFileHandler, get_logger


@pytest.fixture
def tmp_log_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data") / "test.log"


@pytest.fixture
def basic_logger(tmp_log_file: Path) -> Any:
    return get_logger(name="test_basic_logger", filename=str(tmp_log_file))


@pytest.fixture
def logger_with_mask(tmp_log_file: Path) -> Any:
    return get_logger(name="test_logger_with_mask", filename=str(tmp_log_file), mask_sensitive=True)


@pytest.fixture
def logger_with_mask_and_pattarns(tmp_log_file: Path) -> Any:
    return get_logger(
        name="test_logger_with_mask_and_pattarns",
        filename=str(tmp_log_file),
        mask_sensitive=True,
        mask_sensitive_patterns=["pt1", "pt2"],
    )


def test_mask_sensitive(tmp_log_file: Path, logger_with_mask: Any, logger_with_mask_and_pattarns: Any) -> None:
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


def test_existing_logger() -> None:
    logger1 = get_logger(name="test_logger")
    logger2 = get_logger(name="test_logger")
    assert logger1 == logger2


def test_disable_console_logging() -> None:
    logger = get_logger(name="test_logger", console=False)
    assert isinstance(logger, logging.Logger)


def test_levels(tmp_log_file: Path, basic_logger: Any) -> None:
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


def test_duplicate_filter_with_percent_placeholders(tmp_log_file: Path) -> None:
    logger = get_logger(name="test_duplicate_filter_percent", filename=str(tmp_log_file), level=logging.DEBUG)
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


def _make_record(msg: object = "hello world", level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def _fork_worker(
    handler: MultiProcessRotatingFileHandler,
    worker_id: int,
    count: int,
    barrier: multiprocessing.synchronize.Barrier | None = None,
) -> None:
    for idx in range(count - 1):
        handler.emit(_make_record(f"worker={worker_id} idx={idx} {'x' * 40}"))
    if barrier is not None:
        barrier.wait(timeout=10)
    handler.emit(_make_record(f"worker={worker_id} idx={count - 1} {'x' * 40}"))


def _fork_shared_gid_worker(
    handler: MultiProcessRotatingFileHandler,
    worker_id: int,
    count: int,
    barrier: multiprocessing.synchronize.Barrier,
    fallback_ready: multiprocessing.synchronize.Event,
    queue: multiprocessing.Queue[Any],
    deny_primary_lock: bool,
) -> None:
    if deny_primary_lock:
        real_open = os.open

        def open_denied(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == handler._lock_path:
                raise PermissionError("log dir read-only for worker")
            return real_open(path, flags, mode, *args, **kwargs)

        os.open = open_denied  # type: ignore[assignment]
        handler.emit(_make_record(f"worker={worker_id} idx=0 {'x' * 40}"))
        fallback_ready.set()
        start_idx = 1
    else:
        assert fallback_ready.wait(timeout=10)
        start_idx = 0

    for idx in range(start_idx, count - 1):
        handler.emit(_make_record(f"worker={worker_id} idx={idx} {'x' * 40}"))
    barrier.wait(timeout=10)
    handler.emit(_make_record(f"worker={worker_id} idx={count - 1} {'x' * 40}"))
    queue.put((worker_id, handler._active_lock_path))


def _assert_multiprocess_rollover(
    tmp_log_file: Path,
    num_workers: int,
    count: int,
    max_bytes: int,
    max_record_len: int,
) -> list[str]:
    assert (tmp_log_file.parent / "test.log.1").exists()

    backups: list[tuple[int, Path]] = []
    for path in tmp_log_file.parent.glob("test.log.*"):
        if path.name == "test.log.lock":
            continue
        suffix = path.name.removeprefix("test.log.")
        if suffix.isdigit():
            backup_num = int(suffix)
            backup_size = path.stat().st_size
            assert max_bytes - max_record_len <= backup_size <= max_bytes
            backups.append((backup_num, path))

    backups.sort(key=lambda item: item[0], reverse=True)
    chronological_files = [path for _, path in backups] + [tmp_log_file]

    active_lines = tmp_log_file.read_text().splitlines()
    backup_lines_by_file = [path.read_text().splitlines() for _, path in backups]
    for w in range(num_workers):
        final_line = f"worker={w} idx={count - 1} {'x' * 40}"
        assert final_line in active_lines
        for b_lines in backup_lines_by_file:
            assert final_line not in b_lines

    line_re = re.compile(r"^worker=(\d+) idx=(\d+) ")
    last_idx: dict[int, int] = {}
    all_lines: list[str] = []
    for path in chronological_files:
        lines = path.read_text().splitlines()
        all_lines.extend(lines)
        for line in lines:
            if line == "parent-init":
                continue
            match = line_re.match(line)
            assert match is not None, f"Unexpected log line: {line!r}"
            w_id = int(match.group(1))
            idx = int(match.group(2))
            assert idx > last_idx.get(w_id, -1)
            last_idx[w_id] = idx

    return all_lines


def _cleanup_processes(procs: Sequence[multiprocessing.process.BaseProcess]) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)


def test_get_logger_uses_multiprocess_handler(tmp_log_file: Path) -> None:
    log = get_logger(name="test_mp_handler_type", filename=str(tmp_log_file), console=False)
    file_handlers = [h for h in log.handlers if isinstance(h, MultiProcessRotatingFileHandler)]
    assert len(file_handlers) == 1
    file_handlers[0].close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method required",
)
def test_multiprocess_fork_and_rollover(tmp_log_file: Path) -> None:
    num_workers = 4
    count = 15
    max_bytes = 250
    max_record_len = len(f"worker=3 idx={count - 1} {'x' * 40}\n")
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=max_bytes, backupCount=3)
    try:
        # Open the lock FD in parent before forking to verify child PID reopens its own lock FD.
        handler.emit(_make_record("parent-init"))
        parent_lock_fd = handler._lock_fd
        assert parent_lock_fd is not None
        assert handler._lock_pid == os.getpid()

        ctx = multiprocessing.get_context("fork")
        barrier = ctx.Barrier(num_workers, timeout=10)
        procs = [ctx.Process(target=_fork_worker, args=(handler, w, count, barrier)) for w in range(num_workers)]
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

    all_lines = _assert_multiprocess_rollover(tmp_log_file, num_workers, count, max_bytes, max_record_len)
    assert "parent-init" in all_lines or any("worker=" in line for line in all_lines)


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot rename or unlink an open log file")
def test_reopen_if_rotated_and_missing_file(tmp_log_file: Path) -> None:
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


def test_reopen_if_rotated_noop_when_closed_or_invalid_stream(tmp_log_file: Path) -> None:
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


def test_should_rollover_edge_cases(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_do_rollover_skips_below_max_bytes(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=100, backupCount=2)
    try:
        handler.emit(_make_record("small-line"))
        assert 0 < tmp_log_file.stat().st_size < 100
        handler.doRollover()
        assert not tmp_log_file.with_name("test.log.1").exists()
        assert tmp_log_file.read_text() == "small-line\n"
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot rename or unlink an open log file")
def test_do_rollover_skips_when_already_rotated_empty_or_missing(tmp_log_file: Path) -> None:
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


def test_do_rollover_suppresses_file_not_found_error(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=3)
    try:
        handler.emit(_make_record("line-one-long-enough"))
        assert tmp_log_file.stat().st_size >= 20

        monkeypatch.setattr(handler, "rotate", MagicMock(side_effect=FileNotFoundError("vanished")))
        handler.doRollover()
        assert handler.stream is not None
    finally:
        handler.close()


def test_do_rollover_propagates_oserror_on_backup_shift(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_do_rollover_propagates_oserror_on_rotate(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_lock_fallback_and_no_lock_preserves_writes(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = os.open

    def fail_primary_lock(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
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

    def fail_all_locks(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if str(path).endswith(".lock"):
            raise OSError("no lock available")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_all_locks)
    handler_no_lock = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        assert handler_no_lock._get_lock_fd() is None
        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler_no_lock, "handleError", mock_handle_error)
        record = _make_record("rejected-without-lock")
        handler_no_lock.emit(record)
        mock_handle_error.assert_called_once_with(record)
        content = tmp_log_file.read_text()
        assert "written-with-fallback-lock" in content
        assert "rejected-without-lock" not in content
    finally:
        handler_no_lock.close()


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_process_lock_reopens_when_unlinked_or_replaced(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("first"))
        lock_path = tmp_log_file.with_name("test.log.lock")

        # Unlinking lock file triggers FileNotFoundError re-validation and recreates the lock.
        os.unlink(lock_path)
        handler.emit(_make_record("after-unlink"))
        assert lock_path.exists()

        # Replacing lock file while handler holds the old FD guarantees a new inode and re-validation.
        assert handler._lock_fd is not None
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


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method required",
)
def test_multiprocess_lock_unlink_recreate_during_rollover(tmp_log_file: Path) -> None:
    num_workers = 4
    count = 20
    max_bytes = 300
    max_record_len = len(f"worker=3 idx={count - 1} {'x' * 40}\n")
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=max_bytes, backupCount=30)
    lock_path = f"{tmp_log_file}.lock"
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(num_workers, timeout=10)
    procs = [ctx.Process(target=_fork_worker, args=(handler, w, count, barrier)) for w in range(num_workers)]
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

    all_lines = _assert_multiprocess_rollover(tmp_log_file, num_workers, count, max_bytes, max_record_len)
    expected_lines = {f"worker={w} idx={i} {'x' * 40}" for w in range(num_workers) for i in range(count)}
    assert len(all_lines) == len(expected_lines)
    assert set(all_lines) == expected_lines


def test_non_posix_fallback_and_emit_error_handling(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def _inspect_lock_fd_in_child(handler: MultiProcessRotatingFileHandler, queue: multiprocessing.Queue[Any]) -> None:
    queue.put((handler._lock_fd, handler._lock_pid))


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_close_lock_fds_after_fork_closes_registered_handlers(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("open-lock"))
        old_fd = handler._lock_fd
        assert old_fd is not None
        assert handler._lock_pid == os.getpid()

        logger_module._close_lock_fds_after_fork()

        assert handler._lock_fd is None
        assert handler._lock_pid is None
        with pytest.raises(OSError):
            os.fstat(old_fd)
    finally:
        handler.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable on this platform")
def test_idle_forked_child_closes_inherited_lock_fd(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("parent-open-lock"))
        parent_fd = handler._lock_fd
        parent_pid = handler._lock_pid
        assert parent_fd is not None
        assert parent_pid == os.getpid()

        ctx = multiprocessing.get_context("fork")
        queue: multiprocessing.Queue[Any] = ctx.Queue()
        proc = ctx.Process(target=_inspect_lock_fd_in_child, args=(handler, queue))
        try:
            proc.start()
            proc.join(timeout=10)
            assert proc.exitcode == 0
            child_lock_fd, child_lock_pid = queue.get(timeout=5)
        finally:
            _cleanup_processes([proc])
            queue.close()
            queue.join_thread()

        assert child_lock_fd is None
        assert child_lock_pid is None
        assert handler._lock_fd == parent_fd
        assert handler._lock_pid == parent_pid
        assert os.fstat(parent_fd).st_ino == os.stat(handler._active_lock_path).st_ino
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX 0o600 permissions do not apply on Windows")
@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_sidecar_lock_file_has_owner_only_permissions(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("create-lock"))
        lock_path = tmp_log_file.with_name("test.log.lock")
        assert lock_path.exists()
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="Symlinks may require elevated privileges on Windows")
def test_symlink_base_filename_normalized_to_realpath(tmp_log_file: Path) -> None:
    real_log = tmp_log_file.with_name("real.log")
    symlink_log = tmp_log_file.with_name("link.log")
    os.symlink(real_log, symlink_log)

    handler = MultiProcessRotatingFileHandler(filename=symlink_log, maxBytes=1024, backupCount=2)
    try:
        expected_real = os.path.realpath(real_log)
        assert handler.baseFilename == expected_real
        assert handler._lock_path == f"{expected_real}.lock"

        handler.emit(_make_record("via-symlink"))
        assert "via-symlink" in real_log.read_text()
        assert os.path.exists(f"{expected_real}.lock")
        assert not os.path.exists(f"{symlink_log}.lock")
    finally:
        handler.close()


def test_should_rollover_uses_utf8_byte_length(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=2, encoding="utf-8")
    try:
        large_record = _make_record("é" * 10)
        # Empty file (size == 0) never rolls over even when encoded record exceeds maxBytes.
        assert handler.shouldRollover(large_record) is False

        # 2 bytes on disk + 11 chars ("é"*10 + "\n") = 13 < 20, but 2 + 21 UTF-8 bytes = 23 >= 20.
        handler.emit(_make_record("a"))
        assert tmp_log_file.stat().st_size == 2
        assert len("é" * 10 + handler.terminator) + 2 < 20
        assert handler.shouldRollover(large_record) is True
    finally:
        handler.close()


def test_do_rollover_retry_after_rotate_failure_does_not_reshift_backups(
    tmp_log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=3)
    try:
        handler.emit(_make_record("backup-two-long-enough"))
        handler.emit(_make_record("backup-one-long-enough"))
        handler.emit(_make_record("active-log-long-enough"))

        backup_1 = tmp_log_file.with_name("test.log.1")
        backup_2 = tmp_log_file.with_name("test.log.2")
        backup_3 = tmp_log_file.with_name("test.log.3")
        assert backup_1.read_text() == "backup-one-long-enough\n"
        assert backup_2.read_text() == "backup-two-long-enough\n"
        assert tmp_log_file.read_text() == "active-log-long-enough\n"

        real_rotate = handler.rotate
        monkeypatch.setattr(handler, "rotate", MagicMock(side_effect=PermissionError("rotate failed")))
        with pytest.raises(PermissionError, match="rotate failed"):
            handler.doRollover()

        assert not backup_1.exists()
        assert backup_2.read_text() == "backup-one-long-enough\n"
        assert backup_3.read_text() == "backup-two-long-enough\n"
        assert tmp_log_file.read_text() == "active-log-long-enough\n"

        monkeypatch.setattr(handler, "rotate", real_rotate)
        handler.doRollover()

        assert backup_1.read_text() == "active-log-long-enough\n"
        assert backup_2.read_text() == "backup-one-long-enough\n"
        assert backup_3.read_text() == "backup-two-long-enough\n"
        assert tmp_log_file.read_text() == ""
    finally:
        handler.close()


def test_emit_formats_outside_process_lock(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        events: list[str] = []
        in_lock = False
        orig_process_lock = handler._process_lock

        @contextmanager
        def tracking_lock() -> Generator[None, None, None]:
            nonlocal in_lock
            with orig_process_lock():
                in_lock = True
                events.append("lock_enter")
                try:
                    yield
                finally:
                    events.append("lock_exit")
                    in_lock = False

        class LazyMessage:
            def __str__(self) -> str:
                assert not in_lock
                events.append("format")
                return "formatted-before-lock"

        monkeypatch.setattr(handler, "_process_lock", tracking_lock)
        handler.emit(_make_record(LazyMessage()))

        assert events == ["format", "lock_enter", "lock_exit"]
        assert tmp_log_file.read_text() == "formatted-before-lock\n"
    finally:
        handler.close()


def test_step_and_hash_levels(tmp_log_file: Path) -> None:
    log: Any = get_logger(name="test_step_hash_logger", filename=str(tmp_log_file), console=False)
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


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_lock_mode_and_primary_sidecar_no_split(
    tmp_log_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tmp = tmp_path / "fallback_tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))

    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        os.chmod(tmp_log_file, 0o644)
        assert handler._lock_mode() == 0o600

        os.chmod(tmp_log_file, 0o660)
        assert handler._lock_mode() == 0o660
        fd_gid = handler._get_lock_fd()
        assert fd_gid is not None
        assert stat.S_IMODE(os.stat(handler._lock_path).st_mode) == 0o660
        handler._close_lock_fd()
        os.unlink(handler._lock_path)

        os.chmod(tmp_log_file, 0o666)
        assert handler._lock_mode() == 0o666

        # Existing sidecar lock where O_RDWR raises PermissionError falls back to O_RDONLY.
        lock_path = tmp_log_file.with_name("test.log.lock")
        lock_path.write_text("")
        real_open = os.open

        def open_rdwr_denied(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == handler._lock_path and (flags & os.O_RDWR) and not (flags & os.O_CREAT):
                raise PermissionError("read-only lock")
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_rdwr_denied)
        fd = handler._get_lock_fd()
        assert fd is not None
        assert handler._active_lock_path == handler._lock_path
        handler.emit(_make_record("rdonly-lock-ok"))
        assert "rdonly-lock-ok" in tmp_log_file.read_text()
        handler._close_lock_fd()

        # Existing sidecar lock that cannot be opened at all raises PermissionError when maxBytes > 0
        # without falling back to tempdir, and emit() routes it to handleError.
        temp_opens: list[str] = []

        def open_primary_denied(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == handler._lock_path and not (flags & os.O_CREAT):
                raise PermissionError("cannot open lock")
            if str(path).startswith(str(fake_tmp)):
                temp_opens.append(str(path))
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_primary_denied)
        with pytest.raises(PermissionError, match="cannot open lock"):
            handler._get_lock_fd()
        assert temp_opens == []
        assert list(fake_tmp.iterdir()) == []

        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler, "handleError", mock_handle_error)
        denied_record = _make_record("unwritable-lock-denied")
        handler.emit(denied_record)
        mock_handle_error.assert_called_once_with(denied_record)
        assert "unwritable-lock-denied" not in tmp_log_file.read_text()
    finally:
        handler.close()


def test_append_only_max_bytes_zero_skips_locks(
    tmp_log_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tmp = tmp_path / "fallback_tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))

    real_open = os.open

    def deny_all_locks(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if str(path).endswith(".lock"):
            raise PermissionError("lock creation denied")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_all_locks)

    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=0, backupCount=2)
    try:
        handler.emit(_make_record("append-only-record"))
        assert "append-only-record" in tmp_log_file.read_text()
        assert not tmp_log_file.with_name("test.log.lock").exists()
        assert list(fake_tmp.iterdir()) == []
        assert handler._lock_fd is None
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX 0o660 permissions do not apply on Windows")
@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_shared_gid_fallback_convergence(
    tmp_log_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tmp = tmp_path / "fallback_tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))

    tmp_log_file.touch()
    os.chmod(tmp_log_file, 0o660)
    gid = os.stat(tmp_log_file).st_gid
    real_open = os.open

    def deny_primary_lock(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if str(path) == f"{tmp_log_file}.lock":
            raise PermissionError("log dir read-only for writer B")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_primary_lock)
    writer_b = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    writer_a = None
    try:
        writer_b.emit(_make_record("from-writer-b"))
        expected_dir = fake_tmp / f"simple-logger-gid-{gid}"
        assert writer_b._lock_fd is not None
        assert writer_b._active_lock_path.startswith(str(expected_dir))
        assert stat.S_IMODE(os.stat(expected_dir).st_mode) == 0o770
        assert stat.S_IMODE(os.stat(writer_b._active_lock_path).st_mode) == 0o660
        assert not tmp_log_file.with_name("test.log.lock").exists()

        # Restore primary directory access for Writer A; it must join Writer B's shared fallback lock.
        monkeypatch.setattr(os, "open", real_open)
        writer_a = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
        fd_a = writer_a._get_lock_fd()
        assert fd_a is not None
        assert writer_a._active_lock_path == writer_b._active_lock_path
        assert not tmp_log_file.with_name("test.log.lock").exists()

        writer_a.emit(_make_record("from-writer-a"))
        assert not tmp_log_file.with_name("test.log.lock").exists()
        content = tmp_log_file.read_text()
        assert "from-writer-b" in content
        assert "from-writer-a" in content
    finally:
        if writer_a is not None:
            writer_a.close()
        writer_b.close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method required",
)
@pytest.mark.skipif(os.name == "nt", reason="POSIX 0o660 permissions do not apply on Windows")
@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_multiprocess_fork_shared_gid_fallback_convergence(
    tmp_log_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tmp = tmp_path / "fallback_tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))

    tmp_log_file.touch()
    os.chmod(tmp_log_file, 0o660)
    gid = os.stat(tmp_log_file).st_gid

    num_workers = 2
    count = 15
    max_bytes = 250
    max_record_len = len(f"worker={num_workers - 1} idx={count - 1} {'x' * 40}\n")

    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=max_bytes, backupCount=15)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(num_workers, timeout=10)
    fallback_ready = ctx.Event()
    queue: multiprocessing.Queue[Any] = ctx.Queue()
    procs = [
        ctx.Process(
            target=_fork_shared_gid_worker,
            args=(handler, w, count, barrier, fallback_ready, queue, w == 0),
        )
        for w in range(num_workers)
    ]
    try:
        try:
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=15)
                assert proc.exitcode == 0

            lock_paths = dict(queue.get(timeout=5) for _ in range(num_workers))
        finally:
            _cleanup_processes(procs)
            queue.close()
            queue.join_thread()
    finally:
        handler.close()

    assert len(lock_paths) == num_workers
    assert len(set(lock_paths.values())) == 1
    shared_lock_path = next(iter(lock_paths.values()))
    assert shared_lock_path.startswith(str(fake_tmp / f"simple-logger-gid-{gid}"))
    assert not tmp_log_file.with_name("test.log.lock").exists()

    all_lines = _assert_multiprocess_rollover(tmp_log_file, num_workers, count, max_bytes, max_record_len)
    expected_lines = {f"worker={w} idx={i} {'x' * 40}" for w in range(num_workers) for i in range(count)}
    assert len(all_lines) == len(expected_lines)
    assert set(all_lines) == expected_lines


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_secure_fallback_dir_and_file_validation(
    tmp_log_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tmp = tmp_path / "fallback_tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))

    real_open = os.open
    real_lstat = os.lstat
    real_fstat = os.fstat

    def deny_primary_lock(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if str(path) == f"{tmp_log_file}.lock":
            raise PermissionError("primary dir read-only")
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_primary_lock)

    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        gid = os.stat(tmp_log_file).st_gid
        owner_dir_name = f"simple-logger-{uid}" if hasattr(os, "getuid") else "simple-logger"
        owner_fallback_dir = str(fake_tmp / owner_dir_name)

        def fake_stat_result(st_mode: int, st_uid: int, st_gid: int = gid) -> os.stat_result:
            return os.stat_result((st_mode, 1, 1, 1, st_uid, st_gid, 0, 0, 0, 0))

        # 1. Owner-only fallback_dir lstat OSError and validation failures.
        monkeypatch.setattr(os, "lstat", MagicMock(side_effect=OSError("lstat failed")))
        assert handler._get_lock_fd() is None

        for bad_mode, bad_uid in [
            (stat.S_IFDIR | 0o700, uid + 1),
            (stat.S_IFDIR | 0o755, uid),
            (stat.S_IFLNK | 0o700, uid),
        ]:
            monkeypatch.setattr(
                os,
                "lstat",
                lambda p, m=bad_mode, u=bad_uid: (
                    fake_stat_result(m, u) if str(p) == owner_fallback_dir else real_lstat(p)
                ),
            )
            assert handler._get_lock_fd() is None

        monkeypatch.setattr(os, "lstat", real_lstat)

        # 2. Owner-only fallback_path PermissionError without existing file, then O_RDONLY fallback when it exists.
        def deny_fallback_create(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == f"{tmp_log_file}.lock" or str(path).startswith(owner_fallback_dir):
                raise PermissionError("cannot create fallback lock")
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", deny_fallback_create)
        assert handler._get_lock_fd() is None

        monkeypatch.setattr(os, "open", deny_primary_lock)
        assert handler._get_lock_fd() is not None
        owner_fallback_path = handler._active_lock_path
        handler._close_lock_fd()

        def deny_fallback_rdwr(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == f"{tmp_log_file}.lock":
                raise PermissionError("primary dir read-only")
            if str(path) == owner_fallback_path and (flags & os.O_RDWR):
                raise PermissionError("fallback rdwr denied")
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", deny_fallback_rdwr)
        assert handler._get_lock_fd() is not None
        handler._close_lock_fd()

        monkeypatch.setattr(os, "open", deny_primary_lock)

        # 3. Owner-only fallback_path fstat failures: not a regular file, wrong uid, and OSError.
        for bad_fstat in [
            fake_stat_result(stat.S_IFIFO | 0o600, uid),
            fake_stat_result(stat.S_IFREG | 0o600, uid + 1),
        ]:
            monkeypatch.setattr(os, "fstat", lambda _fd, res=bad_fstat: res)
            assert handler._get_lock_fd() is None

        monkeypatch.setattr(os, "fstat", MagicMock(side_effect=OSError("fstat failed")))
        assert handler._get_lock_fd() is None
        mock_handle_error = MagicMock()
        monkeypatch.setattr(handler, "handleError", mock_handle_error)
        rejected_record = _make_record("rejected-while-fstat-fails")
        handler.emit(rejected_record)
        mock_handle_error.assert_called_once_with(rejected_record)

        monkeypatch.setattr(os, "fstat", real_fstat)

        # 4. Group-shared (simple-logger-gid-<gid>) validation failures and O_RDONLY fallback.
        os.chmod(tmp_log_file, 0o660)
        gid_fallback_dir = str(fake_tmp / f"simple-logger-gid-{gid}")

        for bad_mode, bad_uid, bad_gid in [
            (stat.S_IFDIR | 0o770, uid, gid + 1),
            (stat.S_IFDIR | 0o770, uid + 1, gid),
            (stat.S_IFDIR | 0o777, uid, gid),
            (stat.S_IFLNK | 0o770, uid, gid),
        ]:
            monkeypatch.setattr(
                os,
                "lstat",
                lambda p, m=bad_mode, u=bad_uid, g=bad_gid: (
                    fake_stat_result(m, u, g) if str(p) == gid_fallback_dir else real_lstat(p)
                ),
            )
            assert handler._get_lock_fd() is None

        monkeypatch.setattr(os, "lstat", real_lstat)

        for bad_fstat in [
            fake_stat_result(stat.S_IFIFO | 0o660, uid, gid),
            fake_stat_result(stat.S_IFREG | 0o660, uid, gid + 1),
            fake_stat_result(stat.S_IFREG | 0o660, uid + 1, gid),
        ]:
            monkeypatch.setattr(os, "fstat", lambda _fd, res=bad_fstat: res)
            assert handler._get_lock_fd() is None

        monkeypatch.setattr(os, "fstat", real_fstat)
        assert handler._get_lock_fd() is not None
        gid_fallback_path = handler._active_lock_path
        handler._close_lock_fd()

        def deny_gid_fallback_rdwr(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if str(path) == f"{tmp_log_file}.lock":
                raise PermissionError("primary dir read-only")
            if str(path) == gid_fallback_path and (flags & os.O_RDWR):
                raise PermissionError("gid fallback rdwr denied")
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", deny_gid_fallback_rdwr)
        assert handler._get_lock_fd() is not None
        handler._close_lock_fd()

        monkeypatch.setattr(os, "open", deny_primary_lock)
        handler.emit(_make_record("still-writes-after-rejected-fallback"))
        assert "still-writes-after-rejected-fallback" in tmp_log_file.read_text()
    finally:
        handler.close()


@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_flock_unsupported_and_unexpected_oserror(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        real_flock = logger_module.fcntl.flock

        def flock_eopnotsupp(fd: int, op: int) -> None:
            if op == logger_module.fcntl.LOCK_EX:
                raise OSError(errno.EOPNOTSUPP, "flock unsupported")
            real_flock(fd, op)

        monkeypatch.setattr(logger_module.fcntl, "flock", flock_eopnotsupp)
        handler.emit(_make_record("written-when-flock-unsupported"))
        assert handler._lock_fd is None
        assert handler._lock_pid is None
        assert "written-when-flock-unsupported" in tmp_log_file.read_text()

        for err_no, err_msg in ((errno.EIO, "unexpected io error"), (errno.ENOLCK, "lock limit exhausted")):

            def flock_err(fd: int, op: int, _err_no: int = err_no, _err_msg: str = err_msg) -> None:
                if op == logger_module.fcntl.LOCK_EX:
                    raise OSError(_err_no, _err_msg)
                real_flock(fd, op)

            monkeypatch.setattr(logger_module.fcntl, "flock", flock_err)
            mock_handle_error = MagicMock()
            monkeypatch.setattr(handler, "handleError", mock_handle_error)
            record = _make_record(f"flock-error-{err_no}")
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)
            assert f"flock-error-{err_no}" not in tmp_log_file.read_text()
    finally:
        handler.close()


def test_should_rollover_utf16_no_extra_bom(tmp_log_file: Path) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=13, backupCount=2, encoding="utf-16")
    try:
        # "a\n" with UTF-16 writes 2-byte BOM + 4 bytes = 6 bytes on disk.
        handler.emit(_make_record("a"))
        assert tmp_log_file.stat().st_size == 6

        # "bc\n" is 6 bytes without BOM (total 12 < 13); counting an extra 2-byte BOM would give 14 >= 13.
        assert handler.shouldRollover(_make_record("bc")) is False
        handler.maxBytes = 12
        assert handler.shouldRollover(_make_record("bc")) is True
    finally:
        handler.close()


def test_should_rollover_crlf_expansion(tmp_log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=16, backupCount=2, encoding="utf-8")
    try:
        handler.emit(_make_record("ab"))
        assert tmp_log_file.stat().st_size == 3

        monkeypatch.setattr(os, "linesep", "\r\n")
        # "1\n2\n3\n4\n5\n" is 10 bytes with LF (3+10=13 < 16), but 15 bytes with CRLF (3+15=18 >= 16).
        assert handler.shouldRollover(_make_record("1\n2\n3\n4\n5")) is True
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX 0o660 permissions do not apply on Windows")
def test_log_acl_mid_rotation_fstat_and_cached_fallback(
    tmp_log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_log_file.touch()
    os.chmod(tmp_log_file, 0o660)

    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2, delay=True)
    try:
        # Before any stat succeeds, missing file and no stream returns default without caching.
        real_stat = os.stat
        monkeypatch.setattr(os, "stat", MagicMock(side_effect=FileNotFoundError("missing")))
        assert handler._log_acl() == (0o600, None, None)
        assert handler._cached_acl is None

        # Open stream, then simulate mid-rotation missing baseFilename: falls back to fstat(stream.fileno()).
        monkeypatch.setattr(os, "stat", real_stat)
        expected_stat = os.stat(tmp_log_file)
        expected_acl = (0o660, expected_stat.st_uid, expected_stat.st_gid)
        handler.stream = handler._open()

        monkeypatch.setattr(os, "stat", MagicMock(side_effect=FileNotFoundError("mid-rotation")))
        assert handler._log_acl() == expected_acl
        assert handler._cached_acl == expected_acl

        # If fstat/fileno also fails (e.g. stream closed or None), cached ACL is returned.
        handler.stream.close()
        assert handler._log_acl() == expected_acl
        handler.stream = None
        assert handler._log_acl() == expected_acl
    finally:
        handler.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX 0o660 permissions do not apply on Windows")
@pytest.mark.skipif(logger_module.fcntl is None, reason="POSIX fcntl locking required")
def test_is_trusted_owner_and_sync_primary_lock_acl(
    tmp_log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pwd = MagicMock()
    fake_grp = MagicMock()
    monkeypatch.setattr(logger_module, "pwd", fake_pwd)
    monkeypatch.setattr(logger_module, "grp", fake_grp)

    fake_pwd.getpwuid.return_value = MagicMock(pw_gid=50, pw_name="worker_b")
    assert logger_module._is_trusted_owner(200, 100, 100, 50, group_shared=True) is True

    fake_pwd.getpwuid.return_value = MagicMock(pw_gid=99, pw_name="worker_b")
    fake_grp.getgrgid.return_value = MagicMock(gr_mem=["worker_b"])
    assert logger_module._is_trusted_owner(200, 100, 100, 50, group_shared=True) is True

    fake_grp.getgrgid.return_value = MagicMock(gr_mem=["other"])
    assert logger_module._is_trusted_owner(200, 100, 100, 50, group_shared=True) is False

    fake_pwd.getpwuid.side_effect = KeyError("unknown uid")
    assert logger_module._is_trusted_owner(200, 100, 100, 50, group_shared=True) is False

    tmp_log_file.touch()
    os.chmod(tmp_log_file, 0o600)
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=1024, backupCount=2)
    try:
        handler.emit(_make_record("initial-owner-only"))
        assert stat.S_IMODE(os.stat(handler._lock_path).st_mode) == 0o600

        os.chmod(tmp_log_file, 0o660)
        handler.emit(_make_record("after-group-chmod"))
        assert stat.S_IMODE(os.stat(handler._lock_path).st_mode) == 0o660
    finally:
        handler.close()
