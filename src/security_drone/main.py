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
from security_drone.detection.frame_log import FrameLog
from security_drone.detection.worker import DetectionWorker
from security_drone.mavlink.commands import CommandFailed, send_command, set_message_interval
from security_drone.mavlink.connection import connect
from security_drone.mavlink.mission import PatrolTimer, TriggerWatcher
from security_drone.mavlink.telemetry import parse_heartbeat, parse_position

mavlink = mavutil.mavlink
log = logging.getLogger("security_drone")

HEARTBEAT_LOST_S = 3.0
POSITION_LOG_PERIOD_S = 5.0
FPS_LOG_PERIOD_S = 5.0
# Flight modes that take the vehicle away from the protected area. A ground
# station switching to one of them ends the patrol (the thesis "return" command).
RETURN_MODES = {"RTL", "LAND", "SMART_RTL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Security drone companion app")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--connection", help="override mavlink.connection")
    parser.add_argument("--model", help="override detection.model")
    parser.add_argument("--patrol-time", type=float, help="override mission.patrol_time_s")
    parser.add_argument("--snapshot-dir", help="override detection.snapshot_dir")
    parser.add_argument("--frame-log", help="write one CSV row per processed frame here")
    parser.add_argument("--save-all-detections", action="store_true",
                        help="save every annotated detection frame, e.g. to inspect false positives")
    parser.add_argument("--record-dir",
                        help="save every processed camera frame here, to replay detection offline")
    return parser.parse_args()


class Companion:
    """Main loop: MAVLink telemetry, trigger waypoint, detection and alarm."""

    def __init__(self, master, config, camera, detector, frame_log=None, save_all=False,
                 record_dir=None):
        self.master = master
        self.camera = camera
        self.detector = detector
        self.frame_log = frame_log
        self.save_all = save_all
        self.record_dir = Path(record_dir) if record_dir else None
        self.recorded = 0
        self.detection_config = config["detection"]
        self.watcher = TriggerWatcher(config["mission"]["trigger_seq"])
        self.patrol = PatrolTimer(config["mission"]["patrol_time_s"])
        self.threat_alarm = ThreatAlarm(config["alert"]["clear_after_s"],
                                        config["alert"]["notify_every_s"])
        self.buzzer = LogAlarm()
        self.results = queue.Queue()
        self.worker = None
        self.snapshot_saved = False

        self.state = None
        self.position = None
        self.last_reached_seq = None
        self.detection_started = None
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
            self.check_patrol(now)

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
                    self.patrol.stop()
                    self.stop_detection()
                    self.watcher.reset()
                # Returned from the ground station before the patrol time was up.
                elif self.patrol.running and new_state.mode in RETURN_MODES:
                    log.info("Mode changed to %s: patrol ended", new_state.mode)
                    self.patrol.stop()
                    self.stop_detection()
                self.state = new_state

        elif msg_type == "GLOBAL_POSITION_INT":
            self.position = parse_position(msg)
            if now - self.last_position_log >= POSITION_LOG_PERIOD_S:
                log.info("Position %.7f, %.7f, alt %.1f m", self.position.lat_deg,
                         self.position.lon_deg, self.position.relative_alt_m)
                self.last_position_log = now

        elif msg_type == "MISSION_ITEM_REACHED":
            log.info("Mission item %d reached", msg.seq)
            self.last_reached_seq = msg.seq
            if self.watcher.on_item_reached(msg.seq):
                log.info("Trigger waypoint %d reached: start detection, patrol for %.0f s",
                         msg.seq, self.patrol.duration_s)
                self.start_detection()
                self.patrol.start(now)

    def check_patrol(self, now):
        """Send RTL once the patrol time is up; retried until the autopilot accepts."""
        if not self.patrol.expired(now):
            return
        try:
            send_command(self.master, mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
        except CommandFailed as exc:
            log.error("RTL not accepted, retrying: %s", exc)
            return
        log.info("Patrol time up: RTL accepted")
        send_statustext(self.master, "Patrol done, returning home", mavlink.MAV_SEVERITY_NOTICE)
        self.patrol.stop()
        self.stop_detection()

    def start_detection(self):
        if self.worker is not None:
            return
        self.worker = DetectionWorker(
            self.camera, self.detector, self.results,
            equalize=self.detection_config["equalize"],
            sharpen_edges=self.detection_config["sharpen"],
        )
        self.detection_started = time.monotonic()
        self.worker.start()

    def stop_detection(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.join(timeout=2)
        self.worker = None
        # Drop results still queued, or a frame from the patrol raises an
        # alert after RTL. A late put from the old worker lands in the old queue.
        self.results = queue.Queue()
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
            if self.frame_log is not None:
                alt = self.position.relative_alt_m if self.position else None
                self.frame_log.write(now - self.detection_started, self.last_reached_seq,
                                     alt, result.detections, result.inference_s)
            if self.record_dir is not None:
                self.record_frame(result.frame, now)
            count = len(result.detections)
            if count:
                best = max(d.confidence for d in result.detections)
                log.info("Detected %d person(s), best confidence %.2f", count, best)
                self.save_snapshot(result, now)
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

    def record_frame(self, frame, now):
        """Save the raw frame, named by order, time since detection start and leg."""
        self.record_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.recorded:05d}_t{now - self.detection_started:07.2f}_seq{self.last_reached_seq}.jpg"
        cv2.imwrite(str(self.record_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.recorded += 1

    def save_snapshot(self, result, now):
        """Keep the first annotated detection as evidence (or all with save_all)."""
        if self.snapshot_saved and not self.save_all:
            return
        out_dir = Path(self.detection_config["snapshot_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.save_all:
            # Time since detection start and mission leg, matching the frame log.
            name = f"det_{now - self.detection_started:06.2f}s_seq{self.last_reached_seq}.png"
        else:
            name = "first_detection.png"
        path = out_dir / name
        cv2.imwrite(str(path), draw_detections(result.frame, result.detections))
        if not self.snapshot_saved:
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
    det = config["detection"]
    if args.model:
        det["model"] = args.model
    if args.patrol_time is not None:
        config["mission"]["patrol_time_s"] = args.patrol_time
    if args.snapshot_dir:
        det["snapshot_dir"] = args.snapshot_dir

    # Load the model before flying: it takes a few seconds.
    detector = PersonDetector(det["model"], det["confidence"], det["classes"],
                              det["image_size"], det["device"])
    detector.warm_up()
    log.info("Loaded and warmed up %s", det["model"])
    camera = GazeboCamera(config["camera"]["gazebo_topic"])
    camera.start()

    log.info("Connecting to %s", url)
    master = connect(url, config["mavlink"]["heartbeat_timeout_s"])
    log.info("Heartbeat from system %d component %d",
             master.target_system, master.target_component)

    # The autopilot does not stream position on every link by default.
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    log.info("GLOBAL_POSITION_INT stream requested (ACK accepted)")

    frame_log = FrameLog(args.frame_log) if args.frame_log else None
    companion = Companion(master, config, camera, detector, frame_log,
                          save_all=args.save_all_detections, record_dir=args.record_dir)
    try:
        companion.run()
    except KeyboardInterrupt:
        companion.stop_detection()
        log.info("Stopped")
    finally:
        if frame_log is not None:
            frame_log.close()


if __name__ == "__main__":
    main()
