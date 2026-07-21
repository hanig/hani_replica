#!/usr/bin/env python3
"""Watchdog for the Engram Slack bot launchd service."""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABEL = "com.engram.bot"
DEFAULT_SERVICE_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "bot_healthcheck.log"
DEFAULT_BOT_APP_LOG = PROJECT_ROOT / "logs" / "engram.log"
DEFAULT_BOT_ERROR_LOG = PROJECT_ROOT / "logs" / "bot_error.log"
DEFAULT_EXPECTED_COMMAND = "scripts/run_bot.py"
BROKEN_PIPE_MARKER = "BrokenPipeError"
DEFAULT_LOG_SCAN_BYTES = 2_000_000
SOCKET_FAILURE_MARKERS = (
    BROKEN_PIPE_MARKER,
    "Failed to establish a connection",
    "Failed to retrieve WSS URL",
    "Failed to check the state of sock",
)


@dataclass
class ServiceStatus:
    """Parsed launchd service status."""

    loaded: bool
    running: bool
    pid: int | None = None
    state: str = ""
    error: str = ""


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and capture text output."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _launch_domain(uid: int | None = None) -> str:
    """Return the launchd GUI domain for the current user."""
    return f"gui/{uid if uid is not None else os.getuid()}"


def _parse_launchctl_print(output: str) -> ServiceStatus:
    """Parse enough of `launchctl print` output for watchdog decisions."""
    state = ""
    pid: int | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("state ="):
            state = line.split("=", 1)[1].strip()
        elif line.startswith("pid ="):
            value = line.split("=", 1)[1].strip()
            try:
                pid = int(value)
            except ValueError:
                pid = None

    return ServiceStatus(
        loaded=True,
        running=state in {"running", "active"} and pid is not None,
        pid=pid,
        state=state,
    )


def get_service_status(label: str, uid: int | None = None) -> ServiceStatus:
    """Get launchd service status for a label."""
    result = _run_command(["launchctl", "print", f"{_launch_domain(uid)}/{label}"])
    if result.returncode != 0:
        return ServiceStatus(
            loaded=False,
            running=False,
            error=(result.stderr or result.stdout).strip(),
        )
    return _parse_launchctl_print(result.stdout)


def get_pid_command(pid: int) -> str:
    """Return the command line for a process ID, or an empty string."""
    result = _run_command(["ps", "auxww"])
    if result.returncode != 0:
        return ""
    pid_text = str(pid)
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 10)
        if len(parts) >= 11 and parts[1] == pid_text:
            return parts[10]
    return ""


def is_service_healthy(status: ServiceStatus, expected_command: str | None = None) -> bool:
    """Determine whether the service is healthy enough to leave alone."""
    if not status.loaded or not status.running or status.pid is None:
        return False
    if expected_command:
        return expected_command in get_pid_command(status.pid)
    return True


def _parse_log_timestamp(line: str) -> datetime | None:
    """Parse the timestamp prefix used by the bot logs."""
    if len(line) < 23:
        return None
    try:
        return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def _read_log_tail(log_path: Path, max_bytes: int = DEFAULT_LOG_SCAN_BYTES) -> list[str]:
    """Read a bounded tail of a log file as lines."""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
                f.readline()
            return f.read().decode(errors="replace").splitlines()
    except OSError as e:
        logging.warning("Could not read log file %s: %s", log_path, e)
        return []


def count_recent_log_occurrences(
    log_path: Path,
    marker: str,
    lookback_seconds: int,
    now: datetime | None = None,
) -> int:
    """Count marker occurrences in recent timestamped log lines."""
    if not log_path.exists():
        return 0

    now = now or datetime.now()
    cutoff = now - timedelta(seconds=lookback_seconds)
    count = 0

    lines = _read_log_tail(log_path)

    for line in reversed(lines):
        timestamp = _parse_log_timestamp(line)
        if timestamp is None:
            continue
        if timestamp < cutoff:
            break
        if marker in line:
            count += 1

    return count


def count_recent_log_marker_occurrences(
    log_path: Path,
    markers: tuple[str, ...],
    lookback_seconds: int,
    now: datetime | None = None,
) -> int:
    """Count recent timestamped log lines containing any marker."""
    if not markers:
        return 0

    if not log_path.exists():
        return 0

    now = now or datetime.now()
    cutoff = now - timedelta(seconds=lookback_seconds)
    count = 0

    lines = _read_log_tail(log_path)

    for line in reversed(lines):
        timestamp = _parse_log_timestamp(line)
        if timestamp is None:
            continue
        if timestamp < cutoff:
            break
        if any(marker in line for marker in markers):
            count += 1

    return count


def has_recent_broken_pipe_loop(
    error_log: Path,
    lookback_seconds: int,
    threshold: int,
    now: datetime | None = None,
) -> bool:
    """Return true if recent logs show a repeated Slack broken-pipe loop."""
    if threshold <= 0:
        return False
    return (
        count_recent_log_occurrences(
            error_log,
            BROKEN_PIPE_MARKER,
            lookback_seconds,
            now=now,
        )
        >= threshold
    )


def has_recent_socket_failure_loop(
    log_paths: list[Path],
    lookback_seconds: int,
    threshold: int,
    now: datetime | None = None,
) -> bool:
    """Return true if recent logs show a repeated Slack Socket Mode failure loop."""
    if threshold <= 0:
        return False

    total = sum(
        count_recent_log_marker_occurrences(
            log_path,
            SOCKET_FAILURE_MARKERS,
            lookback_seconds,
            now=now,
        )
        for log_path in log_paths
    )
    return total >= threshold


def repair_service(
    label: str,
    service_plist: Path,
    uid: int | None = None,
    dry_run: bool = False,
) -> bool:
    """Bootstrap if needed, then kickstart the service."""
    domain = _launch_domain(uid)

    if not service_plist.exists():
        logging.error("Service plist does not exist: %s", service_plist)
        return False

    status = get_service_status(label, uid=uid)
    if not status.loaded:
        command = ["launchctl", "bootstrap", domain, str(service_plist)]
        logging.warning("Service is not loaded; bootstrapping with: %s", " ".join(command))
        if not dry_run:
            result = _run_command(command)
            if result.returncode != 0:
                logging.error("Bootstrap failed: %s", (result.stderr or result.stdout).strip())
                return False

    command = ["launchctl", "kickstart", "-k", f"{domain}/{label}"]
    logging.warning("Restarting service with: %s", " ".join(command))
    if dry_run:
        return True

    result = _run_command(command)
    if result.returncode != 0:
        logging.error("Kickstart failed: %s", (result.stderr or result.stdout).strip())
        return False

    return is_service_healthy(get_service_status(label, uid=uid))


def configure_logging(log_file: Path) -> None:
    """Configure watchdog logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def main() -> int:
    """Run the bot health check."""
    parser = argparse.ArgumentParser(description="Check and restart the Engram bot if needed.")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="launchd service label")
    parser.add_argument(
        "--service-plist",
        type=Path,
        default=DEFAULT_SERVICE_PLIST,
        help="Path to the bot launchd plist",
    )
    parser.add_argument(
        "--expected-command",
        default=DEFAULT_EXPECTED_COMMAND,
        help="Substring expected in the bot process command line",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument(
        "--app-log",
        type=Path,
        default=DEFAULT_BOT_APP_LOG,
        help="Bot application log to inspect for repeated Socket Mode failures",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_BOT_ERROR_LOG,
        help="Bot stderr log to inspect for repeated Socket Mode failures",
    )
    parser.add_argument(
        "--broken-pipe-lookback-seconds",
        type=int,
        default=60,
        help="Window for counting recent BrokenPipeError entries",
    )
    parser.add_argument(
        "--broken-pipe-threshold",
        type=int,
        default=5,
        help="Restart if at least this many recent BrokenPipeError entries are found",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log actions without changing launchd state")
    args = parser.parse_args()

    configure_logging(args.log_file)

    status = get_service_status(args.label)
    service_healthy = is_service_healthy(status, expected_command=args.expected_command)
    socket_failure_loop = has_recent_socket_failure_loop(
        [args.error_log, args.app_log],
        args.broken_pipe_lookback_seconds,
        args.broken_pipe_threshold,
    )

    if service_healthy and not socket_failure_loop:
        logging.info("Service %s is healthy: state=%s pid=%s", args.label, status.state, status.pid)
        return 0

    if socket_failure_loop:
        logging.warning(
            "Service %s has a recent Socket Mode failure loop in %s and %s",
            args.label,
            args.error_log,
            args.app_log,
        )

    logging.warning(
        "Service %s unhealthy: loaded=%s running=%s state=%s pid=%s error=%s",
        args.label,
        status.loaded,
        status.running,
        status.state,
        status.pid,
        status.error,
    )

    if repair_service(args.label, args.service_plist, dry_run=args.dry_run):
        logging.info("Service %s repaired successfully", args.label)
        return 0

    logging.error("Service %s could not be repaired", args.label)
    return 1


if __name__ == "__main__":
    sys.exit(main())
