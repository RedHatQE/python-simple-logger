import logging
import multiprocessing
import os
import re
from datetime import datetime
from itertools import pairwise
from multiprocessing.synchronize import Barrier
from pathlib import Path
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


def test_get_logger_uses_multiprocess_handler(tmp_log_file):
    log = get_logger(name="test_mp_handler_type", filename=tmp_log_file, console=False)
    file_handlers = [h for h in log.handlers if isinstance(h, MultiProcessRotatingFileHandler)]
    assert len(file_handlers) == 1
    file_handlers[0].close()


def test_multiprocess_fork_and_rollover(tmp_log_file):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=250, backupCount=3)
    # Open the lock FD in parent before forking to verify child PID reopens its own lock FD.
    handler.emit(_make_record("parent-init"))
    parent_lock_fd = handler._lock_fd
    assert parent_lock_fd is not None
    assert handler._lock_pid == os.getpid()

    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_fork_worker, args=(handler, w, 15)) for w in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=15)
        assert proc.exitcode == 0

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
        tmp_log_file.write_text("from-other-process\n")
        handler.doRollover()
        assert tmp_log_file.read_text() == "from-other-process\n"
    finally:
        handler.close()


def test_do_rollover_suppresses_transient_oserror(tmp_log_file, monkeypatch):
    handler = MultiProcessRotatingFileHandler(filename=tmp_log_file, maxBytes=20, backupCount=2)
    try:
        handler.emit(_make_record("line-one-long-enough"))
        handler.emit(_make_record("line-two-long-enough"))
        assert tmp_log_file.with_name("test.log.1").exists()

        monkeypatch.setattr(handler, "rotate", MagicMock(side_effect=OSError("transient")))
        handler.doRollover()
        assert handler.stream is not None
    finally:
        handler.close()


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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_E2E_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\S+)\s+\S+\s+INFO\s+worker=(\d+)\s+msg=(\d{3})$")


def _spawn_get_logger_worker(
    log_file: str,
    logger_name: str,
    worker_id: int,
    num_messages: int,
    file_max_bytes: int,
    file_backup_count: int,
    start_barrier: Barrier,
    final_barrier: Barrier,
) -> None:
    log = get_logger(
        name=logger_name,
        filename=log_file,
        file_max_bytes=file_max_bytes,
        file_backup_count=file_backup_count,
        console=False,
        duplicate_filter=False,
    )
    try:
        start_barrier.wait(timeout=30)
        for idx in range(num_messages - 1):
            log.info(f"worker={worker_id} msg={idx:03d}")
        # Synchronize before the final message so all processes emit msg=059 after prior rollovers settle.
        final_barrier.wait(timeout=30)
        log.info(f"worker={worker_id} msg={num_messages - 1:03d}")
    finally:
        for handler in list(log.handlers):
            handler.close()


def _fork_get_logger_worker(
    logger_name: str,
    worker_id: int,
    num_messages: int,
    start_barrier: Barrier,
    final_barrier: Barrier,
) -> None:
    log = logging.getLogger(logger_name)
    start_barrier.wait(timeout=30)
    for idx in range(num_messages - 1):
        log.info(f"worker={worker_id} msg={idx:03d}")
    final_barrier.wait(timeout=30)
    log.info(f"worker={worker_id} msg={num_messages - 1:03d}")


def _assert_e2e_multiprocess_logs(
    log_file: Path,
    num_workers: int,
    num_messages: int,
    file_max_bytes: int,
    file_backup_count: int,
    *,
    expect_all_messages: bool,
    expect_parent_init: bool = False,
) -> None:
    assert log_file.exists()
    active_size = log_file.stat().st_size
    assert 0 < active_size <= file_max_bytes

    backups: list[Path] = []
    for idx in range(1, file_backup_count + 1):
        backup = log_file.with_name(f"{log_file.name}.{idx}")
        if backup.exists():
            backups.append(backup)
        else:
            break

    # Ensure rollover actually occurred and there are no gaps or unexpected higher-numbered backups.
    assert len(backups) >= 1
    for idx in range(len(backups) + 1, file_backup_count + 2):
        assert not log_file.with_name(f"{log_file.name}.{idx}").exists()

    for backup in backups:
        size = backup.stat().st_size
        assert file_max_bytes // 2 <= size <= file_max_bytes, f"{backup.name} size {size} out of expected range"

    # Chronological order: oldest backup (.N) -> ... -> .1 -> active log_file.
    ordered_files = [*reversed(backups), log_file]
    last_idx_by_worker: dict[int, int] = {}
    seen_pairs: set[tuple[int, int]] = set()
    parent_init_count = 0
    file_mean_timestamps: list[float] = []
    file_max_timestamps: list[datetime] = []
    final_msg_idx = num_messages - 1
    active_final_workers: set[int] = set()
    backup_final_workers: set[int] = set()

    for path in ordered_files:
        file_timestamps: list[datetime] = []
        for raw_line in path.read_text().splitlines():
            clean_line = _ANSI_RE.sub("", raw_line)
            if expect_parent_init and clean_line.endswith(" parent-init"):
                parent_init_count += 1
                continue
            match = _E2E_LINE_RE.match(clean_line)
            assert match is not None, f"Corrupted or unexpected log line in {path.name}: {raw_line!r}"
            ts = datetime.fromisoformat(match.group(1))
            worker_id = int(match.group(2))
            msg_idx = int(match.group(3))
            assert 0 <= worker_id < num_workers
            assert 0 <= msg_idx < num_messages

            pair = (worker_id, msg_idx)
            assert pair not in seen_pairs, f"Duplicate message {pair} in {path.name}"
            seen_pairs.add(pair)

            prev_idx = last_idx_by_worker.get(worker_id, -1)
            assert msg_idx > prev_idx, (
                f"Out-of-order message for worker={worker_id} in {path.name}: {msg_idx} <= {prev_idx}"
            )
            last_idx_by_worker[worker_id] = msg_idx

            file_timestamps.append(ts)
            if msg_idx == final_msg_idx:
                if path == log_file:
                    active_final_workers.add(worker_id)
                else:
                    backup_final_workers.add(worker_id)

        assert file_timestamps, f"No worker messages found in {path.name}"
        epoch_secs = [ts.timestamp() for ts in file_timestamps]
        file_mean_timestamps.append(sum(epoch_secs) / len(epoch_secs))
        file_max_timestamps.append(max(file_timestamps))

    assert active_final_workers == set(range(num_workers))
    assert backup_final_workers == set()

    # Chronological ordering across backups (.N -> ... -> .1 -> app.log): timestamps must advance.
    for older_mean_ts, newer_mean_ts in pairwise(file_mean_timestamps):
        assert older_mean_ts <= newer_mean_ts
    for older_ts, newer_ts in pairwise(file_max_timestamps):
        assert older_ts <= newer_ts

    if expect_parent_init:
        assert parent_init_count == 1
    if expect_all_messages:
        expected_pairs = {(w, m) for w in range(num_workers) for m in range(num_messages)}
        assert seen_pairs == expected_pairs


def test_get_logger_multiprocess_spawn_e2e(tmp_path):
    log_file = tmp_path / "app.log"
    num_workers = 4
    num_messages = 60
    file_max_bytes = 2000
    file_backup_count = 5

    ctx = multiprocessing.get_context("spawn")
    start_barrier = ctx.Barrier(num_workers)
    final_barrier = ctx.Barrier(num_workers)
    procs = [
        ctx.Process(
            target=_spawn_get_logger_worker,
            args=(
                str(log_file),
                "mp_spawn",
                worker_id,
                num_messages,
                file_max_bytes,
                file_backup_count,
                start_barrier,
                final_barrier,
            ),
        )
        for worker_id in range(num_workers)
    ]

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    _assert_e2e_multiprocess_logs(
        log_file=log_file,
        num_workers=num_workers,
        num_messages=num_messages,
        file_max_bytes=file_max_bytes,
        file_backup_count=file_backup_count,
        expect_all_messages=False,
    )


def test_get_logger_multiprocess_fork_e2e(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method not supported on this platform")

    log_file = tmp_path / "app.log"
    logger_name = "mp_fork"
    num_workers = 4
    num_messages = 60
    file_max_bytes = 2000
    file_backup_count = 10

    log = get_logger(
        name=logger_name,
        filename=str(log_file),
        file_max_bytes=file_max_bytes,
        file_backup_count=file_backup_count,
        console=False,
        duplicate_filter=False,
    )
    try:
        # Emit before forking so parent already holds open stream and lock FD.
        log.info("parent-init")

        ctx = multiprocessing.get_context("fork")
        start_barrier = ctx.Barrier(num_workers)
        final_barrier = ctx.Barrier(num_workers)
        procs = [
            ctx.Process(
                target=_fork_get_logger_worker,
                args=(logger_name, worker_id, num_messages, start_barrier, final_barrier),
            )
            for worker_id in range(num_workers)
        ]

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=30)
            assert proc.exitcode == 0
    finally:
        for handler in list(log.handlers):
            handler.close()
        logger_module.LOGGERS.pop(logger_name, None)

    _assert_e2e_multiprocess_logs(
        log_file=log_file,
        num_workers=num_workers,
        num_messages=num_messages,
        file_max_bytes=file_max_bytes,
        file_backup_count=file_backup_count,
        expect_all_messages=True,
        expect_parent_init=True,
    )


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
