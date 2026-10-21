import argparse
import datetime
import logging
import queue
import time
from pathlib import Path

import cv2
from pymavlink import mavutil

from security_drone.alert.alarm import AlarmEvent, LogAlarm, ThreatAlarm
from security_drone.alert.statustext import send_statustext, threat_text
from security_drone.camera.gazebo_camera import GazeboCamera
from security_drone.config import DEFAULT_CONFIG_PATH, load_config
from security_drone.detection.detector import PersonDetector, draw_detections
from security_drone.detection.worker import DetectionWorker
from security_drone.mavlink.commands import set_message_interval
from security_drone.mavlink.connection import connect
from security_drone.mavlink.mission import TriggerWatcher
from security_drone.mavlink.telemetry import parse_heartbeat, parse_position

mavlink = mavutil.mavlink
log = logging.getLogger("security_drone")

HEARTBEAT_LOST_S = 3.0
POSITION_LOG_PERIOD_S = 5.0
FPS_LOG_PERIOD_S = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="Security drone companion app")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--connection", help="override mavlink.connection")
    return parser.parse_args()


class Companion:
    """Main loop: MAVLink telemetry, trigger waypoint, detection and alarm."""

    def __init__(self, master, config, camera, detector):
        self.master = master
        self.camera = camera
        self.detector = detector
        self.detection_config = config["detection"]
        self.watcher = TriggerWatcher(config["mission"]["trigger_seq"])
        self.threat_alarm = ThreatAlarm(config["alert"]["clear_after_s"],
                                        config["alert"]["notify_every_s"])
        self.buzzer = LogAlarm()
        self.results = queue.Queue()
        self.worker = None
        self.snapshot_saved = False

        self.state = None
        self.position = None
        self.last_heartbeat = time.monotonic()
        self.heartbeat_lost = False
        self.last_position_log = 0.0
        self.inference_times = []
        self.last_fps_log = time.monotonic()

    def run(self):
        while True:
            # Short timeout so detection results are handled promptly.
            msg = self.master.recv_match(
                type=["HEARTBEAT", "GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED"],
                blocking=True,
                timeout=0.1,
            )
            now = time.monotonic()
            self.check_heartbeat(now)
            if msg is not None:
                self.handle_message(msg, now)
            self.handle_results(now)

    def check_heartbeat(self, now):
        if now - self.last_heartbeat > HEARTBEAT_LOST_S and not self.heartbeat_lost:
            log.warning("No heartbeat for %.0f s", now - self.last_heartbeat)
            self.heartbeat_lost = True

    def handle_message(self, msg, now):
        # Ignore traffic that is not from our autopilot, e.g. GCS heartbeats
        # the autopilot forwards between its links.
        if msg.get_srcSystem() != self.master.target_system:
            return
        if msg.get_srcComponent() != self.master.target_component:
            return

        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            self.last_heartbeat = now
            if self.heartbeat_lost:
                log.info("Heartbeat back")
                self.heartbeat_lost = False
            new_state = parse_heartbeat(msg)
            if new_state != self.state:
                log.info("Mode %s, %s", new_state.mode,
                         "ARMED" if new_state.armed else "disarmed")
                # Landed: stop looking and be ready for the next flight.
                if self.state is not None and self.state.armed and not new_state.armed:
                    self.stop_detection()
                    self.watcher.reset()
                self.state = new_state

        elif msg_type == "GLOBAL_POSITION_INT":
            self.position = parse_position(msg)
            if now - self.last_position_log >= POSITION_LOG_PERIOD_S:
                log.info("Position %.7f, %.7f, alt %.1f m", self.position.lat_deg,
                         self.position.lon_deg, self.position.relative_alt_m)
                self.last_position_log = now

        elif msg_type == "MISSION_ITEM_REACHED":
            log.info("Mission item %d reached", msg.seq)
            if self.watcher.on_item_reached(msg.seq):
                log.info("Trigger waypoint %d reached: start detection", msg.seq)
                self.start_detection()

    def start_detection(self):
        if self.worker is not None:
            return
        self.worker = DetectionWorker(
            self.camera, self.detector, self.results,
            equalize=self.detection_config["equalize"],
            sharpen_edges=self.detection_config["sharpen"],
        )
        self.worker.start()

    def stop_detection(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.join(timeout=2)
        self.worker = None
        if self.threat_alarm.active:
            self.threat_alarm.active = False
            self.buzzer.off()
        log.info("Detection stopped")

    def handle_results(self, now):
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            self.inference_times.append(result.inference_s)
            count = len(result.detections)
            if count:
                best = max(d.confidence for d in result.detections)
                log.info("Detected %d person(s), best confidence %.2f", count, best)
                self.save_snapshot(result)
            self.handle_alarm(self.threat_alarm.update(count, now), count)
        self.log_fps(now)

    def handle_alarm(self, event, count):
        if event is None:
            return
        if event is AlarmEvent.CLEARED:
            self.buzzer.off()
            send_statustext(self.master, "Threat cleared", mavlink.MAV_SEVERITY_NOTICE)
            log.info("Sent STATUSTEXT: Threat cleared")
            return
        if event is AlarmEvent.ACTIVATED:
            self.buzzer.on()
        lat = self.position.lat_deg if self.position else None
        lon = self.position.lon_deg if self.position else None
        text = threat_text(count, lat, lon, datetime.datetime.now(datetime.timezone.utc))
        send_statustext(self.master, text)
        log.info("Sent STATUSTEXT: %s", text)

    def save_snapshot(self, result):
        """Keep the first annotated detection as evidence, not every frame."""
        if self.snapshot_saved:
            return
        out_dir = Path(self.detection_config["snapshot_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sim_detection_wp2.png"
        cv2.imwrite(str(path), draw_detections(result.frame, result.detections))
        log.info("Saved %s", path)
        self.snapshot_saved = True

    def log_fps(self, now):
        if now - self.last_fps_log < FPS_LOG_PERIOD_S:
            return
        if self.inference_times:
            mean = sum(self.inference_times) / len(self.inference_times)
            log.info("Detection: %d frames, %.1f ms/frame (%.1f FPS)",
                     len(self.inference_times), mean * 1000, 1 / mean)
        self.inference_times.clear()
        self.last_fps_log = now


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    url = args.connection or config["mavlink"]["connection"]

    # Load the model before flying: it takes a few seconds.
    det = config["detection"]
    detector = PersonDetector(det["model"], det["confidence"], det["classes"],
                              det["image_size"], det["device"])
    log.info("Loaded %s", det["model"])
    camera = GazeboCamera(config["camera"]["gazebo_topic"])
    camera.start()

    log.info("Connecting to %s", url)
    master = connect(url, config["mavlink"]["heartbeat_timeout_s"])
    log.info("Heartbeat from system %d component %d",
             master.target_system, master.target_component)

    # The autopilot does not stream position on every link by default.
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    log.info("GLOBAL_POSITION_INT stream requested (ACK accepted)")

    companion = Companion(master, config, camera, detector)
    try:
        companion.run()
    except KeyboardInterrupt:
        companion.stop_detection()
        log.info("Stopped")


if __name__ == "__main__":
    main()
