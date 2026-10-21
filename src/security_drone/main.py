import argparse
import logging
import time

from pymavlink import mavutil

from security_drone.config import DEFAULT_CONFIG_PATH, load_config
from security_drone.mavlink.commands import set_message_interval
from security_drone.mavlink.connection import connect
from security_drone.mavlink.mission import TriggerWatcher
from security_drone.mavlink.telemetry import parse_heartbeat, parse_position

mavlink = mavutil.mavlink
log = logging.getLogger("security_drone")

HEARTBEAT_LOST_S = 3.0
POSITION_LOG_PERIOD_S = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="Security drone companion app")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--connection", help="override mavlink.connection")
    return parser.parse_args()


def run(master, trigger_seq):
    watcher = TriggerWatcher(trigger_seq)
    state = None
    last_heartbeat = time.monotonic()
    last_position_log = 0.0
    heartbeat_lost = False

    while True:
        msg = master.recv_match(
            type=["HEARTBEAT", "GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED"],
            blocking=True,
            timeout=1,
        )
        now = time.monotonic()

        if now - last_heartbeat > HEARTBEAT_LOST_S and not heartbeat_lost:
            log.warning("No heartbeat for %.0f s", now - last_heartbeat)
            heartbeat_lost = True

        # Ignore traffic that is not from our autopilot, e.g. GCS heartbeats
        # the autopilot forwards between its links.
        if msg is None or msg.get_srcSystem() != master.target_system:
            continue
        if msg.get_srcComponent() != master.target_component:
            continue

        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            last_heartbeat = now
            if heartbeat_lost:
                log.info("Heartbeat back")
                heartbeat_lost = False
            new_state = parse_heartbeat(msg)
            if new_state != state:
                log.info("Mode %s, %s", new_state.mode,
                         "ARMED" if new_state.armed else "disarmed")
                # A new flight may follow a landing, so watch for the trigger again.
                if state is not None and state.armed and not new_state.armed:
                    watcher.reset()
                state = new_state

        elif msg_type == "GLOBAL_POSITION_INT":
            if now - last_position_log >= POSITION_LOG_PERIOD_S:
                pos = parse_position(msg)
                log.info("Position %.7f, %.7f, alt %.1f m",
                         pos.lat_deg, pos.lon_deg, pos.relative_alt_m)
                last_position_log = now

        elif msg_type == "MISSION_ITEM_REACHED":
            log.info("Mission item %d reached", msg.seq)
            if watcher.on_item_reached(msg.seq):
                log.info("Trigger waypoint %d reached: start detection", msg.seq)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    url = args.connection or config["mavlink"]["connection"]

    log.info("Connecting to %s", url)
    master = connect(url, config["mavlink"]["heartbeat_timeout_s"])
    log.info("Heartbeat from system %d component %d",
             master.target_system, master.target_component)

    # The autopilot does not stream position on every link by default.
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    log.info("GLOBAL_POSITION_INT stream requested (ACK accepted)")

    try:
        run(master, config["mission"]["trigger_seq"])
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()
