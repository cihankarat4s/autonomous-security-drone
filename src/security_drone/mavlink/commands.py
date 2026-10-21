import time

from pymavlink import mavutil

mavlink = mavutil.mavlink


class CommandFailed(RuntimeError):
    """The autopilot rejected a command or did not answer it."""


def result_name(result):
    """Turn a MAV_RESULT number into its name, e.g. 4 -> 'MAV_RESULT_FAILED'."""
    entry = mavlink.enums["MAV_RESULT"].get(result)
    return entry.name if entry else str(result)


def send_command(master, command, params=(), timeout_s=3.0):
    """Send COMMAND_LONG and wait for its COMMAND_ACK.

    Raises CommandFailed unless the result is MAV_RESULT_ACCEPTED.
    In SITL, a takeoff sent after the vehicle had disarmed itself
    (DISARM_DELAY) came back as MAV_RESULT_FAILED, so success is never assumed.
    """
    # COMMAND_LONG always carries exactly 7 float parameters.
    padded = list(params) + [0] * (7 - len(params))
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command,
        0,  # confirmation
        *padded,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = master.recv_match(
            type="COMMAND_ACK",
            blocking=True,
            timeout=deadline - time.monotonic(),
        )
        # The ACK may belong to a command sent by someone else (e.g. the GCS).
        if ack is None or ack.command != command:
            continue
        if ack.result == mavlink.MAV_RESULT_IN_PROGRESS:
            continue
        if ack.result != mavlink.MAV_RESULT_ACCEPTED:
            raise CommandFailed(f"Command {command} rejected: {result_name(ack.result)}")
        return ack
    raise CommandFailed(f"No COMMAND_ACK for command {command} within {timeout_s} s")


def set_message_interval(master, message_id, rate_hz):
    """Ask the autopilot to stream a message at rate_hz on this link."""
    interval_us = 1_000_000 / rate_hz
    return send_command(
        master,
        mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        (message_id, interval_us),
    )
