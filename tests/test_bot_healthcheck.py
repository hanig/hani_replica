"""Tests for the bot health-check watchdog."""

from datetime import datetime
from unittest.mock import patch

from scripts.bot_healthcheck import (
    ServiceStatus,
    _parse_launchctl_print,
    count_recent_log_occurrences,
    has_recent_broken_pipe_loop,
    is_service_healthy,
    repair_service,
)


def test_parse_launchctl_print_running_pid():
    """Parse running launchctl output."""
    status = _parse_launchctl_print(
        """
        state = running
        pid = 12345
        """
    )

    assert status.loaded is True
    assert status.running is True
    assert status.pid == 12345
    assert status.state == "running"


def test_parse_launchctl_print_active_pid():
    """Some launchd services report active with a valid PID."""
    status = _parse_launchctl_print(
        """
        state = active
        pid = 12345
        """
    )

    assert status.loaded is True
    assert status.running is True
    assert status.pid == 12345
    assert status.state == "active"


def test_parse_launchctl_print_missing_pid_is_not_running():
    """A running state without a PID is not healthy."""
    status = _parse_launchctl_print("state = running")

    assert status.loaded is True
    assert status.running is False
    assert status.pid is None


@patch("scripts.bot_healthcheck.get_pid_command")
def test_is_service_healthy_checks_expected_command(mock_command):
    """Expected command substring must match the running process."""
    status = ServiceStatus(loaded=True, running=True, pid=123, state="running")
    mock_command.return_value = "/usr/bin/python /repo/scripts/run_bot.py"

    assert is_service_healthy(status, expected_command="scripts/run_bot.py") is True
    assert is_service_healthy(status, expected_command="other.py") is False


@patch("scripts.bot_healthcheck.get_service_status")
def test_repair_service_refuses_missing_plist(mock_status, tmp_path):
    """Repair should not run launchctl when the service plist is missing."""
    mock_status.return_value = ServiceStatus(loaded=False, running=False)

    assert repair_service("com.test", tmp_path / "missing.plist") is False


def test_count_recent_log_occurrences_uses_timestamp_window(tmp_path):
    """Only recent timestamped marker lines are counted."""
    log_path = tmp_path / "bot_error.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-05-23 12:00:00,000 - slack_bolt.App - ERROR - BrokenPipeError",
                "2026-05-23 12:04:00,000 - slack_bolt.App - ERROR - BrokenPipeError",
                "2026-05-23 12:04:30,000 - slack_bolt.App - ERROR - BrokenPipeError",
                "unparseable BrokenPipeError",
            ]
        )
    )

    assert (
        count_recent_log_occurrences(
            log_path,
            "BrokenPipeError",
            lookback_seconds=120,
            now=datetime(2026, 5, 23, 12, 5, 0),
        )
        == 2
    )


def test_has_recent_broken_pipe_loop_respects_threshold(tmp_path):
    """Recent broken-pipe loops become an unhealthy condition."""
    log_path = tmp_path / "bot_error.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-05-23 12:04:00,000 - slack_bolt.App - ERROR - BrokenPipeError",
                "2026-05-23 12:04:10,000 - slack_bolt.App - ERROR - BrokenPipeError",
            ]
        )
    )

    assert has_recent_broken_pipe_loop(
        log_path,
        lookback_seconds=120,
        threshold=2,
        now=datetime(2026, 5, 23, 12, 5, 0),
    )
    assert not has_recent_broken_pipe_loop(
        log_path,
        lookback_seconds=120,
        threshold=3,
        now=datetime(2026, 5, 23, 12, 5, 0),
    )


@patch("scripts.bot_healthcheck.is_service_healthy")
@patch("scripts.bot_healthcheck._run_command")
@patch("scripts.bot_healthcheck.get_service_status")
def test_repair_service_bootstraps_unloaded_service(
    mock_status,
    mock_run,
    mock_healthy,
    tmp_path,
):
    """Unloaded services are bootstrapped before kickstart."""
    plist = tmp_path / "service.plist"
    plist.write_text("<plist></plist>")
    mock_status.side_effect = [
        ServiceStatus(loaded=False, running=False),
        ServiceStatus(loaded=True, running=True, pid=123),
    ]
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = ""
    mock_healthy.return_value = True

    assert repair_service("com.test", plist, uid=501) is True

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert ["launchctl", "bootstrap", "gui/501", str(plist)] in commands
    assert ["launchctl", "kickstart", "-k", "gui/501/com.test"] in commands
