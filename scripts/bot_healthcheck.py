#!/usr/bin/env python3
"""Watchdog for the Engram Slack bot launchd service."""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABEL = "com.engram.bot"
DEFAULT_SERVICE_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "bot_healthcheck.log"
DEFAULT_EXPECTED_COMMAND = "scripts/run_bot.py"


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
    parser.add_argument("--dry-run", action="store_true", help="Log actions without changing launchd state")
    args = parser.parse_args()

    configure_logging(args.log_file)

    status = get_service_status(args.label)
    if is_service_healthy(status, expected_command=args.expected_command):
        logging.info("Service %s is healthy: state=%s pid=%s", args.label, status.state, status.pid)
        return 0

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
